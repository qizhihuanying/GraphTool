#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理模型文件脚本
删除 outputs/models/ 下的所有 .pt 文件以节省空间
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='清理模型文件以节省空间')
    parser.add_argument('--models-dir', type=str, default='outputs/models', 
                       help='模型文件目录 (默认: outputs/models)')
    parser.add_argument('--dry-run', action='store_true',
                       help='仅显示将要删除的文件，不实际删除')
    parser.add_argument('--pattern', type=str, default='*.pt',
                       help='文件匹配模式 (默认: *.pt)')
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    
    if not models_dir.exists():
        print(f"[INFO] 模型目录不存在: {models_dir}")
        return 0

    # 查找所有匹配的文件
    pt_files = list(models_dir.glob(args.pattern))
    
    if not pt_files:
        print(f"[INFO] 在 {models_dir} 中未找到匹配 {args.pattern} 的文件")
        return 0

    total_size = 0
    for pt_file in pt_files:
        if pt_file.is_file():
            total_size += pt_file.stat().st_size

    print(f"[INFO] 找到 {len(pt_files)} 个文件，总大小: {total_size / (1024*1024):.2f} MB")
    
    if args.dry_run:
        print("[DRY RUN] 将要删除的文件:")
        for pt_file in pt_files:
            if pt_file.is_file():
                size_mb = pt_file.stat().st_size / (1024*1024)
                print(f"  - {pt_file.name} ({size_mb:.2f} MB)")
        print(f"[DRY RUN] 总共将释放: {total_size / (1024*1024):.2f} MB")
        return 0

    # 确认删除
    response = input(f"确认删除 {len(pt_files)} 个文件 ({total_size / (1024*1024):.2f} MB)? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("[INFO] 取消删除操作")
        return 0

    # 执行删除
    deleted_count = 0
    deleted_size = 0
    
    for pt_file in pt_files:
        if pt_file.is_file():
            try:
                file_size = pt_file.stat().st_size
                pt_file.unlink()
                deleted_count += 1
                deleted_size += file_size
                print(f"[DELETED] {pt_file.name}")
            except Exception as e:
                print(f"[ERROR] 删除 {pt_file.name} 失败: {e}")

    print(f"[SUCCESS] 成功删除 {deleted_count} 个文件，释放空间: {deleted_size / (1024*1024):.2f} MB")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
