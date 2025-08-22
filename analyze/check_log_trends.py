#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

EPOCH_LINE_RE = re.compile(
    r"Epoch:\s*(\d+),\s*Train Loss:\s*([0-9.]+),\s*Val Loss:\s*([0-9.]+),\s*Recall:\s*([0-9.]+),\s*F1:\s*([0-9.]+)"
)
TEST_LINE_RE = re.compile(
    r"Test Loss:\s*([0-9.]+),\s*Test Recall:\s*([0-9.]+),\s*Test F1:\s*([0-9.]+)"
)



def parse_log_file(path: Path) -> Tuple[List[int], List[float], List[float], List[float], Tuple[float, float, float] | None]:
    """Parse a single log file and return per-epoch lists and final test metrics.
    Returns (epochs, val_losses, recalls, f1s, test_metrics) where test_metrics is (loss, recall, f1) or None.
    """
    matches: Dict[int, Tuple[float, float, float]] = {}
    test_metrics: Tuple[float, float, float] | None = None
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = EPOCH_LINE_RE.search(line)
                if m:
                    ep = int(m.group(1))
                    # train_loss = float(m.group(2))  # parsed but unused
                    val_loss = float(m.group(3))
                    recall = float(m.group(4))
                    f1 = float(m.group(5))
                    # Keep the last occurrence for an epoch if duplicated
                    matches[ep] = (val_loss, recall, f1)
                t = TEST_LINE_RE.search(line)
                if t:
                    test_loss = float(t.group(1))
                    test_recall = float(t.group(2))
                    test_f1 = float(t.group(3))
                    test_metrics = (test_loss, test_recall, test_f1)
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return [], [], [], [], None

    if not matches:
        return [], [], [], [], test_metrics

    epochs_sorted = sorted(matches.keys())
    val_losses = [matches[e][0] for e in epochs_sorted]
    recalls = [matches[e][1] for e in epochs_sorted]
    f1s = [matches[e][2] for e in epochs_sorted]
    return epochs_sorted, val_losses, recalls, f1s, test_metrics


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
    epochs, val_losses, recalls, f1s, test_metrics = parse_log_file(path)
    result = {
        'file': str(path),
        'epochs': epochs,
        'num_epochs': len(epochs),
        'val_losses': val_losses,
        'recalls': recalls,
        'f1s': f1s,
        'test_metrics': test_metrics,
        'strict_val_loss_dec': False,
        'strict_recall_inc': False,
        'strict_f1_inc': False,
        'nonstrict_val_loss_dec': False,
        'nonstrict_recall_inc': False,
        'nonstrict_f1_inc': False,
        'violations': {}
    }

    if len(epochs) >= 2:
        if strict:
            result['strict_val_loss_dec'] = is_strictly_decreasing(val_losses)
            result['strict_recall_inc'] = is_strictly_increasing(recalls)
            result['strict_f1_inc'] = is_strictly_increasing(f1s)
            if not result['strict_val_loss_dec']:
                idx = first_violation_index(val_losses, lambda a, b: a < b)
                result['violations']['val_loss'] = idx
            if not result['strict_recall_inc']:
                idx = first_violation_index(recalls, lambda a, b: a > b)
                result['violations']['recall'] = idx
            if not result['strict_f1_inc']:
                idx = first_violation_index(f1s, lambda a, b: a > b)
                result['violations']['f1'] = idx
        else:
            result['nonstrict_val_loss_dec'] = is_non_increasing(val_losses)
            result['nonstrict_recall_inc'] = is_non_decreasing(recalls)
            result['nonstrict_f1_inc'] = is_non_decreasing(f1s)
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

    strict = args.strict
    ok_files = []
    best_ok = None  # (test_recall, file_name)

    for lf in log_files:
        res = check_file(lf, strict=strict)
        epochs = res['epochs']
        if len(epochs) < 2:
            print(f"[SKIP] {lf.name}: not enough epochs ({len(epochs)})")
            continue

        val_first, val_last = res['val_losses'][0], res['val_losses'][-1]
        rec_first, rec_last = res['recalls'][0], res['recalls'][-1]
        f1_first, f1_last = res['f1s'][0], res['f1s'][-1]

        if strict:
            ok = res['strict_val_loss_dec'] and res['strict_recall_inc'] and res['strict_f1_inc']
        else:
            ok = res['nonstrict_val_loss_dec'] and res['nonstrict_recall_inc'] and res['nonstrict_f1_inc']

        status = 'OK' if ok else 'FAIL'
        mode = 'strict' if strict else 'nonstrict'
        print(f"[{status}] ({mode}) {lf.name} | epochs={len(epochs)} | Val {val_first:.4f}->{val_last:.4f} | Rec {rec_first:.4f}->{rec_last:.4f} | F1 {f1_first:.4f}->{f1_last:.4f}")

        if not ok and strict:
            v = res['violations']
            viol_msgs = []
            if 'val_loss' in v and v['val_loss'] is not None and v['val_loss'] >= 0:
                viol_msgs.append(f"val_loss breaks at i={v['val_loss']}(epoch {epochs[v['val_loss']]})")
            if 'recall' in v and v['recall'] is not None and v['recall'] >= 0:
                viol_msgs.append(f"recall breaks at i={v['recall']}(epoch {epochs[v['recall']]})")
            if 'f1' in v and v['f1'] is not None and v['f1'] >= 0:
                viol_msgs.append(f"f1 breaks at i={v['f1']}(epoch {epochs[v['f1']]})")
            if viol_msgs:
                print("       Violations: " + "; ".join(viol_msgs))

        if ok:
            ok_files.append(lf.name)
            # 记录最佳 test recall
            tm = res.get('test_metrics')
            if tm is not None:
                test_recall = tm[1]
                if (best_ok is None) or (test_recall > best_ok[0]):
                    best_ok = (test_recall, lf.name)

    print("\nSummary:")
    print(f"  Total logs: {len(log_files)}")
    print(f"  Passing ({'strict' if strict else 'nonstrict'}): {len(ok_files)}")
    if ok_files:
        for name in ok_files:
            print(f"    - {name}")
    if best_ok is not None:
        print(f"\nBest among OK (by Test Recall): {best_ok[1]} (Test Recall={best_ok[0]:.4f})")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

