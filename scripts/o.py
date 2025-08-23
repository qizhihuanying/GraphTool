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
    --dtype     内存占用张量精度 fp16/fp32（默认 fp16）
    --chunk-mb  单块分配大小（MiB，默认 256）
"""

import argparse
import math
import time
import torch


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
    p.add_argument("--gpu", type=int, required=True, help="GPU 编号")
    p.add_argument("--gb", type=float, default=18.0, help="目标占用显存(GB)")
    p.add_argument("--seconds", type=int, default=60000, help="运行时长(秒)")
    p.add_argument("--dtype", type=str, choices=["fp16", "fp32"], default="fp32", help="占用张量精度")
    p.add_argument("--chunk-mb", type=int, default=256, help="单块分配大小(MiB)")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请在有 NVIDIA GPU 的环境运行")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    # 查询初始可用显存
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except Exception:
        # 某些驱动不支持 mem_get_info，退化为仅按目标分配
        free_bytes, total_bytes = 0, 0

    print(f"[GPU {args.gpu}] Total: {bytes_to_human(total_bytes)}, Free: {bytes_to_human(free_bytes)}")

    # 目标显存与安全余量
    target_bytes = int(args.gb * (1024 ** 3))
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

    # 计算负载：根据剩余空闲显存选择合适的矩阵大小
    try:
        free_for_compute, _ = torch.cuda.mem_get_info(device)
    except Exception:
        free_for_compute = 512 * 1024 ** 2  # 保守估计

    # 预留 256MiB 给计算中间结果
    compute_budget = max(free_for_compute - (256 * 1024 ** 2), 64 * 1024 ** 2)
    # 两个矩阵 + 结果约等于 3 * N^2 * itemsize <= compute_budget
    # N 上取整为 256 的倍数（利于 Tensor Core）
    if compute_budget > 0:
        n_est = int(math.sqrt(compute_budget / (3.0 * itemsize)))
        # 规整到 256 的倍数
        if n_est < 256:
            n_est = 256
        n = (n_est // 256) * 256
    else:
        n = 256

    print(f"计算矩阵大小: {n} x {n} (dtype={args.dtype})")

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

