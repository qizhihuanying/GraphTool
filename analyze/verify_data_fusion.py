#!/usr/bin/env python3
"""
数据融合验证脚本
验证预处理数据是否正确融合了G2和G3数据集
"""

import json
import torch
from pathlib import Path
from collections import defaultdict, Counter
import sys
import os

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_json_data(file_path):
    """加载JSON数据文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"❌ 加载文件失败 {file_path}: {e}")
        return []

def analyze_raw_data(data_dir):
    """分析原始G2和G3数据"""
    print("=" * 60)
    print("📊 原始数据分析")
    print("=" * 60)
    
    g2_path = data_dir / "G2_query.json"
    g3_path = data_dir / "G3_query.json"
    
    results = {}
    
    # 分析G2数据
    if g2_path.exists():
        g2_data = load_json_data(g2_path)
        g2_tools = set()
        g2_queries = []
        
        for sample in g2_data:
            if 'query' in sample:
                g2_queries.append(sample['query'])
            if 'api_list' in sample:
                for api in sample['api_list']:
                    tool_id = f"{api['tool_name']}::{api['api_name']}"
                    g2_tools.add(tool_id)
        
        results['G2'] = {
            'samples': len(g2_data),
            'unique_tools': len(g2_tools),
            'queries': len(g2_queries),
            'tools': g2_tools
        }
        
        print(f"📁 G2数据:")
        print(f"   样本数量: {len(g2_data)}")
        print(f"   唯一工具: {len(g2_tools)}")
        print(f"   查询数量: {len(g2_queries)}")
    else:
        print(f"⚠️  G2文件不存在: {g2_path}")
        results['G2'] = None
    
    # 分析G3数据
    if g3_path.exists():
        g3_data = load_json_data(g3_path)
        g3_tools = set()
        g3_queries = []
        
        for sample in g3_data:
            if 'query' in sample:
                g3_queries.append(sample['query'])
            if 'api_list' in sample:
                for api in sample['api_list']:
                    tool_id = f"{api['tool_name']}::{api['api_name']}"
                    g3_tools.add(tool_id)
        
        results['G3'] = {
            'samples': len(g3_data),
            'unique_tools': len(g3_tools),
            'queries': len(g3_queries),
            'tools': g3_tools
        }
        
        print(f"📁 G3数据:")
        print(f"   样本数量: {len(g3_data)}")
        print(f"   唯一工具: {len(g3_tools)}")
        print(f"   查询数量: {len(g3_queries)}")
    else:
        print(f"⚠️  G3文件不存在: {g3_path}")
        results['G3'] = None
    
    # 计算重叠和总计
    if results['G2'] and results['G3']:
        common_tools = results['G2']['tools'] & results['G3']['tools']
        total_unique_tools = results['G2']['tools'] | results['G3']['tools']
        
        print(f"\n🔗 数据重叠分析:")
        print(f"   G2独有工具: {len(results['G2']['tools'] - results['G3']['tools'])}")
        print(f"   G3独有工具: {len(results['G3']['tools'] - results['G2']['tools'])}")
        print(f"   共同工具: {len(common_tools)}")
        print(f"   总唯一工具: {len(total_unique_tools)}")
        print(f"   总样本数: {results['G2']['samples'] + results['G3']['samples']}")
        
        results['overlap'] = {
            'common_tools': len(common_tools),
            'total_unique_tools': len(total_unique_tools),
            'total_samples': results['G2']['samples'] + results['G3']['samples']
        }
    
    return results

def analyze_preprocessed_data(preprocessed_dir):
    """分析预处理数据"""
    print("\n" + "=" * 60)
    print("🔧 预处理数据分析")
    print("=" * 60)
    
    # 检查文件存在性
    files_to_check = [
        'full_graph.pt',
        'training_samples.pt',
        'validation_samples.pt', 
        'test_samples.pt'
    ]
    
    missing_files = []
    for file_name in files_to_check:
        file_path = preprocessed_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        print(f"❌ 缺少预处理文件: {missing_files}")
        return None
    
    results = {}
    
    # 分析完整图数据
    try:
        full_graph = torch.load(preprocessed_dir / 'full_graph.pt', weights_only=True)
        tool_to_idx = full_graph['tool_to_idx']
        edge_index = full_graph['edge_index']
        node_features = full_graph['node_features']
        
        results['graph'] = {
            'num_tools': len(tool_to_idx),
            'num_edges': edge_index.size(1),
            'node_feature_dim': node_features.size(1),
            'tools': set(tool_to_idx.keys())
        }
        
        print(f"📊 完整图数据:")
        print(f"   工具节点数: {len(tool_to_idx)}")
        print(f"   边数量: {edge_index.size(1)}")
        print(f"   节点特征维度: {node_features.size(1)}")
        
    except Exception as e:
        print(f"❌ 加载完整图数据失败: {e}")
        return None
    
    # 分析样本数据
    sample_files = ['training_samples.pt', 'validation_samples.pt', 'test_samples.pt']
    sample_names = ['训练', '验证', '测试']
    
    total_samples = 0
    for file_name, name in zip(sample_files, sample_names):
        try:
            samples = torch.load(preprocessed_dir / file_name, weights_only=True)
            sample_count = len(samples)
            total_samples += sample_count
            
            results[name] = {
                'count': sample_count,
                'samples': samples
            }
            
            print(f"   {name}样本: {sample_count}")
            
        except Exception as e:
            print(f"❌ 加载{name}样本失败: {e}")
            return None
    
    results['total_samples'] = total_samples
    print(f"   总样本数: {total_samples}")
    
    return results

def verify_data_consistency(raw_results, preprocessed_results):
    """验证数据一致性"""
    print("\n" + "=" * 60)
    print("✅ 数据一致性验证")
    print("=" * 60)
    
    if not raw_results or not preprocessed_results:
        print("❌ 无法进行一致性验证，缺少必要数据")
        return False
    
    success = True
    
    # 验证工具数量
    if 'overlap' in raw_results:
        expected_tools = raw_results['overlap']['total_unique_tools']
        actual_tools = preprocessed_results['graph']['num_tools']
        
        print(f"🔧 工具数量验证:")
        print(f"   期望工具数: {expected_tools}")
        print(f"   实际工具数: {actual_tools}")
        
        if expected_tools == actual_tools:
            print("   ✅ 工具数量一致")
        else:
            print("   ❌ 工具数量不一致")
            success = False
    
    # 验证样本数量
    if 'overlap' in raw_results:
        expected_samples = raw_results['overlap']['total_samples']
        actual_samples = preprocessed_results['total_samples']
        
        print(f"\n📊 样本数量验证:")
        print(f"   期望样本数: {expected_samples}")
        print(f"   实际样本数: {actual_samples}")
        
        # 由于数据预处理可能会过滤掉一些无效样本，实际样本数可能小于等于期望样本数
        if actual_samples <= expected_samples:
            print("   ✅ 样本数量合理（预处理可能过滤了无效样本）")
            if actual_samples < expected_samples:
                filtered = expected_samples - actual_samples
                print(f"   ℹ️  过滤了 {filtered} 个样本 ({filtered/expected_samples*100:.1f}%)")
        else:
            print("   ❌ 样本数量异常（实际大于期望）")
            success = False
    
    # 验证工具覆盖
    if raw_results['G2'] and raw_results['G3']:
        raw_tools = raw_results['G2']['tools'] | raw_results['G3']['tools']
        preprocessed_tools = preprocessed_results['graph']['tools']
        
        print(f"\n🔗 工具覆盖验证:")
        missing_tools = raw_tools - preprocessed_tools
        extra_tools = preprocessed_tools - raw_tools
        
        if not missing_tools and not extra_tools:
            print("   ✅ 工具集完全一致")
        else:
            if missing_tools:
                print(f"   ⚠️  缺少工具: {len(missing_tools)} 个")
                if len(missing_tools) <= 5:
                    for tool in list(missing_tools)[:5]:
                        print(f"      - {tool}")
            if extra_tools:
                print(f"   ⚠️  额外工具: {len(extra_tools)} 个")
                if len(extra_tools) <= 5:
                    for tool in list(extra_tools)[:5]:
                        print(f"      - {tool}")
    
    return success

def main():
    """主函数"""
    print("🔍 GraphTool 数据融合验证")
    print("验证预处理数据是否正确融合了G2和G3数据集")
    
    # 设置路径
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "datasets" / "ToolBench"
    preprocessed_dir = data_dir / "preprocessed_data"
    
    print(f"\n📂 数据目录: {data_dir}")
    print(f"📂 预处理目录: {preprocessed_dir}")
    
    # 分析原始数据
    raw_results = analyze_raw_data(data_dir)
    
    # 分析预处理数据
    preprocessed_results = analyze_preprocessed_data(preprocessed_dir)
    
    # 验证一致性
    is_consistent = verify_data_consistency(raw_results, preprocessed_results)
    
    # 总结
    print("\n" + "=" * 60)
    print("📋 验证总结")
    print("=" * 60)
    
    if is_consistent:
        print("✅ 数据融合验证通过！预处理数据正确融合了G2和G3数据集。")
    else:
        print("❌ 数据融合验证失败！请检查数据预处理过程。")
    
    print("\n🎯 建议:")
    print("   - 如果验证通过，可以放心使用当前的预处理数据")
    print("   - 如果验证失败，建议重新运行数据预处理")
    print("   - 可以删除 preprocessed_data 目录后重新运行训练来重新预处理")

if __name__ == "__main__":
    main()
