#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
占用指定 GPU 约 18GB 显存，并进行持续矩阵乘法以提升 GPU 利用率。
使用示例：
    python o.py --gpu 0 --gb 18 --seconds 600
参数：
    --gpu       GPU 编号（必填，例如 0）
    --gb        目标占用显存（GB，默认 18）
    --seconds   运行时长（秒，默认 600）
    --dtype     内存占用张量精度 fp16/fp32（默认 fp32）
    --chunk-mb  单块分配大小（MiB，默认 256）
"""

import argparse
import math
import time
import torch
import random
import threading


def bytes_to_human(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024.0
        i += 1
    return f"{f:.2f} {units[i]}"


def parse_args():
    p = argparse.ArgumentParser()
    # 支持：--gpu 1 2 3 或单卡 --gpu 1；也支持 --auto
    p.add_argument("--gpu", type=int, nargs='*', default=None, help="指定一个或多个GPU编号（或使用 --auto）")
    p.add_argument("--auto", action="store_true", help="自动选择空闲GPU（当空闲显存≥阈值时启动）")
    p.add_argument("--min-free-gb", type=float, default=18.0, help="自动模式的最小空闲显存阈值(GB)")
    p.add_argument("--wait-lock", type=str, default="outputs/hparam_search_running.lock", help="等待锁文件路径；存在则等待释放")
    p.add_argument("--poll-interval", type=int, default=30, help="检测间隔(秒)")
    p.add_argument("--gb", type=float, default=18.0, help="目标占用显存(GB)")
    p.add_argument("--seconds", type=int, default=900000, help="运行时长(秒)")
    p.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp32", help="占用张量精度")
    p.add_argument("--chunk-mb", type=int, default=256, help="单块分配大小(MiB)")
    return p.parse_args()


def pick_gpu(min_free_bytes: int) -> int:
    """返回满足空闲显存阈值的GPU编号；若无满足则返回 -1"""
    num = torch.cuda.device_count()
    best_gpu = -1
    best_free = -1
    for i in range(num):
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free >= min_free_bytes and free > best_free:
            best_free = free
            best_gpu = i
    return best_gpu


def wait_for_lock_release(lock_path: str, poll_interval: int):
    """等待锁文件被删除（搜参结束）"""
    import os
    while os.path.exists(lock_path):
        print(f"检测到锁文件 {lock_path}，等待 {poll_interval}s…")
        time.sleep(poll_interval)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请在有 NVIDIA GPU 的环境运行")

    # 如提供了锁路径，则无论 auto 还是手动 GPU 列表，都等待锁释放后再继续
    if args.wait_lock:
        wait_for_lock_release(args.wait_lock, args.poll_interval)

    # 若启用自动模式，挑选GPU集合
    if args.auto:
        min_free_bytes = int(args.min_free_gb * (1024 ** 3))
        # 自动模式：选择满足阈值的所有 GPU 列表
        picked = []
        for i in range(torch.cuda.device_count()):
            try:
                free, _ = torch.cuda.mem_get_info(i)
                if free >= min_free_bytes:
                    picked.append(i)
            except Exception:
                continue
        if not picked:
            print(f"未找到空闲显存≥{args.min_free_gb}GB 的GPU，{args.poll_interval}s后重试…")
            while not picked:
                time.sleep(args.poll_interval)
                picked = []
                for i in range(torch.cuda.device_count()):
                    try:
                        free, _ = torch.cuda.mem_get_info(i)
                        if free >= min_free_bytes:
                            picked.append(i)
                    except Exception:
                        continue
        args.gpu = picked
        print(f"自动选择GPU: {args.gpu}")
    elif args.gpu is None:
        raise ValueError("请提供 --gpu 列表或使用 --auto 模式")

    # 将单个值归一化为列表
    if isinstance(args.gpu, int):
        gpu_list = [args.gpu]
    else:
        gpu_list = list(dict.fromkeys(args.gpu))  # 去重保持顺序

    print(f"目标GPU列表: {gpu_list}")

    # 并行在每个GPU上占用显存并计算（并发启动）
    threads = []
    for gpu in gpu_list:
        t = threading.Thread(target=run_on_gpu, args=(gpu, args), daemon=True)
        t.start()
        threads.append(t)

    # 等待全部线程结束
    for t in threads:
        t.join()

    print("全部指定GPU任务已完成。")


def run_on_gpu(gpu: int, args):
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    # 查询初始可用显存
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        # 某些驱动不支持 mem_get_info，退化为仅按目标分配
        free_bytes, total_bytes = 0, 0

    print(f"[GPU {args.gpu}] Total: {bytes_to_human(total_bytes)}, Free: {bytes_to_human(free_bytes)}")

    # 目标显存与安全余量（加入随机上浮，≤500MiB）
    base_target_bytes = int(args.gb * (1024 ** 3))
    jitter_bytes = random.randint(0, 500) * 1024 ** 2  # 0~500 MiB
    target_bytes = base_target_bytes + jitter_bytes
    safety_reserve = 512 * 1024 ** 2  # 预留 ~512MiB

    if free_bytes > 0:
        alloc_budget = max(min(target_bytes, free_bytes - safety_reserve), 0)
    else:
        alloc_budget = target_bytes  # 无法获知，尽力分配

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    itemsize = torch.finfo(dtype).bits // 8
    chunk_bytes = int(args.chunk_mb * 1024 ** 2)

    if chunk_bytes < itemsize:
        chunk_bytes = itemsize

    print(
        f"计划分配: {bytes_to_human(alloc_budget)} (目标 {bytes_to_human(target_bytes)}), "
        f"块大小: {bytes_to_human(chunk_bytes)}, dtype={args.dtype}"
    )

    # 尝试逐块分配到预算
    blocks = []
    allocated = 0
    while allocated < alloc_budget:
        remain = alloc_budget - allocated
        this_chunk = min(chunk_bytes, remain)
        elems = max(this_chunk // itemsize, 1)
        try:
            t = torch.empty(int(elems), dtype=dtype, device=device)
            blocks.append(t)
            allocated += int(elems * itemsize)
        except RuntimeError as e:
            print(f"分配失败: {str(e).splitlines()[0]}，停止进一步分配。")
            break

    # 再次打印显存信息
    try:
        free_after, total_after = torch.cuda.mem_get_info(device)
        used = total_after - free_after
        print(
            f"分配完成: 约 {bytes_to_human(allocated)}，当前占用: {bytes_to_human(used)} / {bytes_to_human(total_after)}"
        )
    except Exception:
        print(f"分配完成: 约 {bytes_to_human(allocated)}")

    # ==================== 修改开始 ====================
    # 原有的动态计算逻辑已被注释掉
    # try:
    #     free_for_compute, _ = torch.cuda.mem_get_info(device)
    # except Exception:
    #     free_for_compute = 512 * 1024 ** 2  # 保守估计
    #
    # compute_budget = max(free_for_compute - (256 * 1024 ** 2), 64 * 1024 ** 2)
    # if compute_budget > 0:
    #     n_est = int(math.sqrt(compute_budget / (3.0 * itemsize)))
    #     if n_est < 256:
    #         n_est = 256
    #     n = (n_est // 256) * 256
    # else:
    #     n = 256

    # 直接将矩阵大小 n 设置为固定的 2048
    n = 2048
    # ==================== 修改结束 ====================

    print(f"[GPU {gpu}] 计算矩阵大小: {n} x {n} (dtype={args.dtype})")

    # 构建计算张量
    with torch.no_grad():
        try:
            A = torch.randn((n, n), dtype=dtype, device=device)
            B = torch.randn((n, n), dtype=dtype, device=device)
        except RuntimeError as e:
            print(f"计算张量分配失败: {str(e).splitlines()[0]}，降级到更小尺寸…")
            n = 512
            A = torch.randn((n, n), dtype=dtype, device=device)
            B = torch.randn((n, n), dtype=dtype, device=device)

        # 持续计算以拉高利用率
        print(f"开始计算，持续 {args.seconds} 秒，Ctrl+C 可提前结束…")
        end_t = time.time() + args.seconds
        iters = 0
        try:
            while time.time() < end_t:
                # 多次 matmul，期间插入非线性，避免编译一次后优化为空
                C = A @ B
                A = torch.tanh(C)
                C = A @ B
                B = torch.tanh(C)
                torch.cuda.synchronize()
                iters += 2

        except KeyboardInterrupt:
            print("收到中断，结束计算…")

    print("完成。进程即将退出，显存将被自动释放。")


if __name__ == "__main__":
    main()