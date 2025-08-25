#!/usr/bin/env python3
"""
分析日志文件，找出最佳的测试参数配置
"""

import re
import os
from pathlib import Path
from collections import defaultdict
import argparse

def parse_log_file(log_path):
    """解析单个日志文件，提取配置和测试结果"""
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ 读取日志文件失败 {log_path}: {e}")
        return None
    
    # 提取配置信息
    config = {}
    
    # 从文件名提取配置（格式：lr=1e-4_l2=1e-5_type=GAT_layer=10.log）
    filename = log_path.stem
    config_parts = filename.split('_')
    for part in config_parts:
        if '=' in part:
            key, value = part.split('=', 1)
            config[key] = value
    
    # 提取测试结果
    test_results = {}
    
    # 查找测试结果部分（新格式：Test Loss: 0.6279, R@3: 0.6395, R@5: 0.8435, N@3: 0.6587, N@5: 0.7512, C@3: 0.3298, C@5: 0.6686）
    test_pattern = r'Test Loss:\s*([\d.]+),\s*R@3:\s*([\d.]+),\s*R@5:\s*([\d.]+),\s*N@3:\s*([\d.]+),\s*N@5:\s*([\d.]+),\s*C@3:\s*([\d.]+),\s*C@5:\s*([\d.]+)'

    match = re.search(test_pattern, content)
    if match:
        test_results = {
            'Test_Loss': float(match.group(1)),
            'R@3': float(match.group(2)),
            'R@5': float(match.group(3)),
            'N@3': float(match.group(4)),
            'N@5': float(match.group(5)),
            'C@3': float(match.group(6)),
            'C@5': float(match.group(7))
        }
    
    # 提取训练信息
    training_info = {}
    
    # 提取最终验证损失
    val_loss_pattern = r'最终验证损失:\s*([\d.]+)'
    val_loss_match = re.search(val_loss_pattern, content)
    if val_loss_match:
        training_info['final_val_loss'] = float(val_loss_match.group(1))
    
    # 提取训练时间
    time_pattern = r'训练总时间:\s*([\d.]+)\s*秒'
    time_match = re.search(time_pattern, content)
    if time_match:
        training_info['training_time'] = float(time_match.group(1))
    
    if not test_results:
        return None
    
    return {
        'config': config,
        'test_results': test_results,
        'training_info': training_info,
        'log_file': str(log_path)
    }

def analyze_logs(logs_dir, dataset_filter=None):
    """分析指定目录下的所有日志文件"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"❌ 日志目录不存在: {logs_dir}")
        return []
    
    results = []
    log_files = list(logs_path.glob("*.log"))
    
    if not log_files:
        print(f"⚠️  在 {logs_dir} 中未找到日志文件")
        return []
    
    print(f"📁 分析目录: {logs_dir}")
    print(f"📄 找到 {len(log_files)} 个日志文件")
    
    for log_file in log_files:
        # 如果指定了数据集过滤器，检查文件名
        if dataset_filter and dataset_filter not in str(log_file):
            continue
            
        result = parse_log_file(log_file)
        if result:
            results.append(result)
        else:
            print(f"⚠️  跳过无效日志: {log_file.name}")
    
    return results

def find_best_configs(results, metric='R@5', top_k=5):
    """找出指定指标的最佳配置"""
    if not results:
        return []
    
    # 按指定指标排序
    sorted_results = sorted(results, key=lambda x: x['test_results'].get(metric, 0), reverse=True)
    
    return sorted_results[:top_k]

def print_results(results, metric='R@5', title="最佳配置"):
    """打印结果"""
    print(f"\n{'='*60}")
    print(f"🏆 {title} (按 {metric} 排序)")
    print(f"{'='*60}")
    
    if not results:
        print("❌ 没有找到有效结果")
        return
    
    for i, result in enumerate(results, 1):
        config = result['config']
        test_results = result['test_results']
        training_info = result['training_info']
        
        print(f"\n🥇 第 {i} 名:")
        print(f"   📊 {metric}: {test_results.get(metric, 'N/A'):.4f}")
        
        # 打印配置
        print(f"   ⚙️  配置:")
        for key, value in config.items():
            print(f"      {key}: {value}")
        
        # 打印所有测试指标
        print(f"   📈 测试结果:")
        for key, value in test_results.items():
            print(f"      {key}: {value:.4f}")
        
        # 打印训练信息
        if training_info:
            print(f"   🕐 训练信息:")
            for key, value in training_info.items():
                if key == 'training_time':
                    print(f"      {key}: {value:.1f}s")
                else:
                    print(f"      {key}: {value:.4f}")
        
        print(f"   📄 日志文件: {Path(result['log_file']).name}")

def main():
    parser = argparse.ArgumentParser(description="分析训练日志，找出最佳配置")
    parser.add_argument('--logs-dir', type=str, default='outputs/logs',
                       help='日志目录路径 (默认: outputs/logs)')
    parser.add_argument('--dataset', type=str,
                       help='数据集标识：l2, l3, l2+l3 或完整的目录名（如 model_name+l2）')
    parser.add_argument('--metric', type=str, default='R@5',
                       choices=['R@3', 'R@5', 'N@3', 'N@5', 'C@3', 'C@5', 'Test_Loss'],
                       help='排序指标 (默认: R@5)')
    parser.add_argument('--top-k', type=int, default=5,
                       help='显示前K个最佳配置 (默认: 5)')

    args = parser.parse_args()
    
    print("🔍 GraphTool 日志分析 - 寻找最佳配置")

    # 根据dataset参数确定实际的日志目录
    if args.dataset:
        # 首先尝试直接作为目录名
        actual_logs_dir = os.path.join(args.logs_dir, args.dataset)

        # 如果直接目录不存在，尝试查找包含该数据集标识的目录
        if not os.path.exists(actual_logs_dir):
            # 列出所有子目录，查找包含数据集标识的目录
            if os.path.exists(args.logs_dir):
                subdirs = [d for d in os.listdir(args.logs_dir)
                          if os.path.isdir(os.path.join(args.logs_dir, d)) and args.dataset in d]

                if len(subdirs) == 1:
                    actual_logs_dir = os.path.join(args.logs_dir, subdirs[0])
                    print(f"🔍 找到匹配目录: {subdirs[0]}")
                elif len(subdirs) > 1:
                    print(f"❌ 找到多个匹配目录: {subdirs}")
                    print(f"💡 请指定完整的目录名，例如: --dataset {subdirs[0]}")
                    return
                else:
                    print(f"❌ 未找到包含 '{args.dataset}' 的目录")
                    print(f"💡 可用目录: {[d for d in os.listdir(args.logs_dir) if os.path.isdir(os.path.join(args.logs_dir, d))]}")
                    return
            else:
                print(f"❌ 日志根目录不存在: {args.logs_dir}")
                return

        logs_dir_to_use = actual_logs_dir
        dataset_filter = None  # 不需要额外过滤，因为已经在正确目录了
    else:
        logs_dir_to_use = args.logs_dir
        dataset_filter = args.dataset

    # 分析日志
    results = analyze_logs(logs_dir_to_use, dataset_filter)
    
    if not results:
        print("❌ 没有找到有效的日志结果")
        return
    
    print(f"✅ 成功解析 {len(results)} 个有效日志")
    
    # 找出最佳配置
    best_configs = find_best_configs(results, args.metric, args.top_k)
    
    # 打印结果
    dataset_info = f" ({args.dataset}数据集)" if args.dataset else ""
    print_results(best_configs, args.metric, f"最佳配置{dataset_info}")
    
    # 额外分析：不同指标的最佳配置
    print(f"\n{'='*60}")
    print("📊 不同指标的最佳单个配置")
    print(f"{'='*60}")
    
    key_metrics = ['R@3', 'R@5', 'N@3', 'N@5', 'C@3', 'C@5']
    for metric in key_metrics:
        best_single = find_best_configs(results, metric, 1)
        if best_single:
            result = best_single[0]
            config_str = ", ".join([f"{k}={v}" for k, v in result['config'].items()])
            print(f"🎯 {metric}: {result['test_results'][metric]:.4f} | {config_str}")

if __name__ == "__main__":
    main()
