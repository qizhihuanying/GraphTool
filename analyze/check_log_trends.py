#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

EPOCH_LINE_RE = re.compile(
    r"Epoch:\s*(\d+),\s*Train Loss:\s*([0-9.]+),\s*Val Loss:\s*([0-9.]+)\s*\(Free:\s*([0-9.]+)\),\s*Recall:\s*([0-9.]+),\s*F1:\s*([0-9.]+)"
)
TEST_LINE_RE = re.compile(
    r"Test Loss:\s*([0-9.]+)\s*\(Free:\s*([0-9.]+)\),\s*Test Recall:\s*([0-9.]+),\s*Test F1:\s*([0-9.]+)"
)



def parse_log_file(path: Path) -> Tuple[List[int], List[float], List[float], List[float], List[float], Tuple[float, float, float, float] | None]:
    """Parse a single log file and return per-epoch lists and final test metrics.
    Returns (epochs, val_losses, val_losses_free, recalls, f1s, test_metrics) where test_metrics is (loss, loss_free, recall, f1) or None.
    """
    matches: Dict[int, Tuple[float, float, float, float]] = {}  # (val_loss, val_loss_free, recall, f1)
    test_metrics: Tuple[float, float, float, float] | None = None
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = EPOCH_LINE_RE.search(line)
                if m:
                    ep = int(m.group(1))
                    # train_loss = float(m.group(2))  # parsed but unused
                    val_loss = float(m.group(3))
                    val_loss_free = float(m.group(4))
                    recall = float(m.group(5))
                    f1 = float(m.group(6))
                    # Keep the last occurrence for an epoch if duplicated
                    matches[ep] = (val_loss, val_loss_free, recall, f1)
                t = TEST_LINE_RE.search(line)
                if t:
                    test_loss = float(t.group(1))
                    test_loss_free = float(t.group(2))
                    test_recall = float(t.group(3))
                    test_f1 = float(t.group(4))
                    test_metrics = (test_loss, test_loss_free, test_recall, test_f1)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return [], [], [], [], [], None

    if not matches:
        return [], [], [], [], [], test_metrics

    epochs_sorted = sorted(matches.keys())
    val_losses = [matches[e][0] for e in epochs_sorted]
    val_losses_free = [matches[e][1] for e in epochs_sorted]
    recalls = [matches[e][2] for e in epochs_sorted]
    f1s = [matches[e][3] for e in epochs_sorted]
    return epochs_sorted, val_losses, val_losses_free, recalls, f1s, test_metrics


def is_strictly_decreasing(xs: List[float]) -> bool:
    return all(xs[i] < xs[i-1] for i in range(1, len(xs)))


def is_non_increasing(xs: List[float]) -> bool:
    return all(xs[i] <= xs[i-1] for i in range(1, len(xs)))


def is_strictly_increasing(xs: List[float]) -> bool:
    return all(xs[i] > xs[i-1] for i in range(1, len(xs)))


def is_non_decreasing(xs: List[float]) -> bool:
    return all(xs[i] >= xs[i-1] for i in range(1, len(xs)))


def first_violation_index(xs: List[float], cmp) -> int:
    """Return the index i where xs[i] violates the relation cmp(xs[i], xs[i-1]).
    If no violation, return -1.
    """
    for i in range(1, len(xs)):
        if not cmp(xs[i], xs[i-1]):
            return i
    return -1


def check_file(path: Path, strict: bool = True) -> Dict:
    epochs, val_losses, val_losses_free, recalls, f1s, test_metrics = parse_log_file(path)
    result = {
        'file': str(path),
        'epochs': epochs,
        'num_epochs': len(epochs),
        'val_losses': val_losses,
        'val_losses_free': val_losses_free,
        'recalls': recalls,
        'f1s': f1s,
        'test_metrics': test_metrics,
        'strict_val_loss_free_dec': False,
        'strict_recall_inc': False,
        'strict_f1_inc': False,
        'nonstrict_val_loss_free_dec': False,
        'nonstrict_recall_inc': False,
        'nonstrict_f1_inc': False,
        'violations': {}
    }

    if len(epochs) >= 2:
        if strict:
            result['strict_val_loss_free_dec'] = is_strictly_decreasing(val_losses_free)
            result['strict_recall_inc'] = is_strictly_increasing(recalls)
            result['strict_f1_inc'] = is_strictly_increasing(f1s)
            if not result['strict_val_loss_free_dec']:
                idx = first_violation_index(val_losses_free, lambda a, b: a < b)
                result['violations']['val_loss_free'] = idx
            if not result['strict_recall_inc']:
                idx = first_violation_index(recalls, lambda a, b: a > b)
                result['violations']['recall'] = idx
            if not result['strict_f1_inc']:
                idx = first_violation_index(f1s, lambda a, b: a > b)
                result['violations']['f1'] = idx
        else:
            result['nonstrict_val_loss_free_dec'] = is_non_increasing(val_losses_free)
            result['nonstrict_recall_inc'] = is_non_decreasing(recalls)
            result['nonstrict_f1_inc'] = is_non_decreasing(f1s)
    return result


def main():
    parser = argparse.ArgumentParser(description='分析 margin 日志：筛选“FREE LOSS最低时 Recall 也是最高”的日志，输出对应的 Test Recall 与最佳参数配置')
    parser.add_argument('--logs-dir', type=str, default='outputs/logs/margin', help='日志目录 (包含 .log)')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"[ERR] 日志目录不存在: {logs_dir}")
        return 1

    log_files = sorted([p for p in logs_dir.glob('*.log') if p.is_file()])
    if not log_files:
        print(f"[WARN] 未在 {logs_dir} 找到 .log 文件")
        return 0

    matched = []  # 收集满足条件的 (file, config_info, min_ep, min_free, recall_at_min_free, test_recall)
    best_overall = None  # (test_recall, file, config_str)

    for lf in log_files:
        ep, val_losses, val_free, recalls, f1s, test_metrics = parse_log_file(lf)
        if not ep:
            continue

        # FREE LOSS 最低值与其所有出现位置
        min_free_value = min(val_free)
        idx_min_list = [i for i, v in enumerate(val_free) if abs(v - min_free_value) < 1e-12]

        # 该日志的最高 Recall
        max_recall = max(recalls) if recalls else None
        if max_recall is None:
            continue

        # 在所有 free_min 的位置中，查找是否存在 recall==max_recall 的位置
        chosen_idx = None
        for i in idx_min_list:
            if abs(recalls[i] - max_recall) < 1e-12:
                chosen_idx = i
                break
        if chosen_idx is None:
            continue  # 不满足条件

        min_ep = ep[chosen_idx]
        min_free = val_free[chosen_idx]
        min_free_recall = recalls[chosen_idx]

        # 提取参数配置
        config_info = ""
        if "margin=" in lf.name:
            parts = lf.name.replace('.log', '').split('_')
            cfg = []
            for p in parts:
                if any(k in p for k in ['lr=', 'l2=', 'type=', 'layer=', 'margin=']):
                    cfg.append(p)
            config_info = ' | '.join(cfg)

        test_recall = None
        if test_metrics is not None:
            test_recall = test_metrics[2]

        matched.append((lf.name, config_info, min_ep, min_free, min_free_recall, test_recall))

        # 维护满足条件集合中的“最佳 Test Recall”
        if test_recall is not None and ((best_overall is None) or (test_recall > best_overall[0])):
            best_overall = (test_recall, lf.name, config_info)

    # 数量统计
    print(f"满足条件的日志数: {len(matched)} / {len(log_files)}")
    if not matched:
        print("未找到满足条件的日志。")
        return 0

    # 输出满足条件的日志（单行输出）
    for name, cfg, ep_min, free_min, rec_at_min, test_rec in matched:
        test_rec_str = f"{test_rec:.4f}" if test_rec is not None else "N/A"
        print(f"file={name} | config={cfg} | epoch_min={ep_min} | free_loss_min={free_min:.4f} | recall_at_min(max)={rec_at_min:.4f} | test_recall={test_rec_str}")

    # 输出满足条件集合中 Test Recall 最高者（单行）
    if best_overall is not None:
        print("=" * 100)
        print(f"BEST | file={best_overall[1]} | config={best_overall[2]} | test_recall={best_overall[0]:.4f}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

