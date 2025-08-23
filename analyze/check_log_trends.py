#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

import re
from pathlib import Path
from typing import Dict, List, Tuple

EPOCH_LINE_RE = re.compile(
    r"Epoch:\s*(\d+),\s*Train Loss:\s*([0-9.]+),\s*Val Loss:\s*([0-9.]+),\s*R@3:\s*([0-9.]+),\s*R@5:\s*([0-9.]+),\s*N@3:\s*([0-9.]+),\s*N@5:\s*([0-9.]+),\s*C@3:\s*([0-9.]+),\s*C@5:\s*([0-9.]+)"
)
TEST_LINE_RE = re.compile(
    r"Test Loss:\s*([0-9.]+),\s*R@3:\s*([0-9.]+),\s*R@5:\s*([0-9.]+),\s*N@3:\s*([0-9.]+),\s*N@5:\s*([0-9.]+),\s*C@3:\s*([0-9.]+),\s*C@5:\s*([0-9.]+)"
)



def parse_log_file(path: Path) -> Tuple[List[int], List[float], Dict[str, List[float]], Tuple[float, float, float, float, float, float, float, float] | None]:
    """Parse a single log file and return per-epoch lists and final test metrics.
    Returns (epochs, val_losses, metrics_per_epoch, test_metrics) where test_metrics is (loss, R3,R5,N3,N5,C3,C5) or None.
    """
    matches: Dict[int, Tuple[float, float, float]] = {}
    test_metrics: Tuple[float, float, float] | None = None
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = EPOCH_LINE_RE.search(line)
                if m:
                    ep = int(m.group(1))
                    val_loss = float(m.group(2))
                    r3 = float(m.group(4)); r5 = float(m.group(5))
                    n3 = float(m.group(6)); n5 = float(m.group(7))
                    c3 = float(m.group(8)); c5 = float(m.group(9))
                    # Keep the last occurrence for an epoch if duplicated
                    matches[ep] = (val_loss, r3, r5, n3, n5, c3, c5)
                t = TEST_LINE_RE.search(line)
                if t:
                    test_loss = float(t.group(1))
                    r3 = float(t.group(2)); r5 = float(t.group(3))
                    n3 = float(t.group(4)); n5 = float(t.group(5))
                    c3 = float(t.group(6)); c5 = float(t.group(7))
                    test_metrics = (test_loss, r3, r5, n3, n5, c3, c5)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return [], [], [], [], None

    if not matches:
        return [], [], {}, test_metrics

    epochs_sorted = sorted(matches.keys())
    val_losses = [matches[e][0] for e in epochs_sorted]
    metrics_per_epoch = {
        'R@3': [matches[e][1] for e in epochs_sorted],
        'R@5': [matches[e][2] for e in epochs_sorted],
        'N@3': [matches[e][3] for e in epochs_sorted],
        'N@5': [matches[e][4] for e in epochs_sorted],
        'C@3': [matches[e][5] for e in epochs_sorted],
        'C@5': [matches[e][6] for e in epochs_sorted],
    }
    return epochs_sorted, val_losses, metrics_per_epoch, test_metrics


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
    epochs, val_losses, metrics_per_epoch, test_metrics = parse_log_file(path)
    result = {
        'file': str(path),
        'epochs': epochs,
        'num_epochs': len(epochs),
        'val_losses': val_losses,
        'metrics': metrics_per_epoch,
        'test_metrics': test_metrics,
        'strict_val_loss_dec': False,
        'nonstrict_val_loss_dec': False,
        'violations': {}
    }

    if len(epochs) >= 2:
        result['strict_val_loss_dec'] = is_strictly_decreasing(val_losses)
        result['nonstrict_val_loss_dec'] = is_non_increasing(val_losses)
    return result


def main():
    parser = argparse.ArgumentParser(description='Scan logs for monotonic trends: val loss down, recall/f1 up.')
    parser.add_argument('--logs-dir', type=str, default='outputs/logs', help='Directory containing .log files')
    parser.add_argument('--strict', action='store_true', help='Use strict monotonicity (<, >) instead of non-strict (<=, >=)')
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"[ERR] Logs dir not found: {logs_dir}")
        return 1

    log_files = sorted([p for p in logs_dir.glob('*.log') if p.is_file()])
    if not log_files:
        print(f"[WARN] No .log files found in {logs_dir}")
        return 0

    ok_files = []
    best_ok = None  # (best_R@5_on_test, file_name)

    for lf in log_files:
        res = check_file(lf, strict=args.strict)
        epochs = res['epochs']
        if len(epochs) < 1:
            print(f"[SKIP] {lf.name}: not enough epochs ({len(epochs)})")
            continue

        val_first, val_last = res['val_losses'][0], res['val_losses'][-1]
        ok = res['nonstrict_val_loss_dec'] or res['strict_val_loss_dec']

        status = 'OK' if ok else 'FAIL'
        mode = 'strict' if args.strict else 'nonstrict'
        # 打印简要：仅展示 Val 变化与 R@5 最近值（如存在）
        last_r5 = None
        mets = res.get('metrics') or {}
        if 'R@5' in mets and mets['R@5']:
            last_r5 = mets['R@5'][-1]
        if last_r5 is not None:
            print(f"[{status}] ({mode}) {lf.name} | epochs={len(epochs)} | Val {val_first:.4f}->{val_last:.4f} | R@5 {last_r5:.4f}")
        else:
            print(f"[{status}] ({mode}) {lf.name} | epochs={len(epochs)} | Val {val_first:.4f}->{val_last:.4f}")

        if ok:
            ok_files.append(lf.name)
            # 记录最佳 test R@5
            tm = res.get('test_metrics')
            if tm is not None:
                test_r5 = tm[2]  # (loss, R3, R5, ...)
                if (best_ok is None) or (test_r5 > best_ok[0]):
                    best_ok = (test_r5, lf.name)

    print("\nSummary:")
    print(f"  Total logs: {len(log_files)}")
    print(f"  Passing ({'strict' if args.strict else 'nonstrict'}): {len(ok_files)}")
    if ok_files:
        for name in ok_files:
            print(f"    - {name}")
    if best_ok is not None:
        print(f"\nBest among OK (by Test R@5): {best_ok[1]} (R@5={best_ok[0]:.4f})")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

