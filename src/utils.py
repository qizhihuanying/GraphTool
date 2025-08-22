"""
工具函数模块：日志记录、指标计算、早停等实用功能
"""
import logging
import numpy as np
import torch
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
import os

def setup_logging(config, log_name: str = "debug") -> logging.Logger:
    """设置日志记录"""
    # 检查是否有环境变量指定的日志文件路径（用于超参数搜索）
    extra_log_file = os.environ.get('EXTRA_LOG_FILE')
    if extra_log_file:
        log_path = Path(extra_log_file)
    else:
        log_path = config['LOGS_DIR'] / f"{log_name}.log"

    # 创建logger
    logger = logging.getLogger(log_name)
    logger.setLevel(getattr(logging, config['LOG_LEVEL']))

    # 避免重复添加handler
    if not logger.handlers:
        # 文件handler
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, config['LOG_LEVEL']))

        # 控制台handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, config['LOG_LEVEL']))

        # 格式化器
        formatter = logging.Formatter(config['LOG_FORMAT'])
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)



    return logger

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算召回率指标

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        metrics: 包含召回率的字典
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 处理空数组情况
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'recall': 0.0
        }

    # 计算召回率
    try:
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)

        return {
            'recall': float(recall)
        }
    except Exception as e:
        logging.warning(f"计算指标时出错: {e}")
        return {
            'recall': 0.0
        }

def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """计算详细的分类指标"""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

class EarlyStopping:
    """早停机制"""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        初始化早停机制

        Args:
            patience: 耐心值，连续多少个epoch没有改善就停止
            min_delta: 最小改善幅度
            mode: 'min'表示指标越小越好，'max'表示指标越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停

        Args:
            score: 当前指标值

        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_better(self, current: float, best: float) -> bool:
        """判断当前分数是否比最佳分数更好"""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

def plot_training_history(train_history: Dict[str, List[float]],
                         val_history: Dict[str, List[float]],
                         save_path: Path = None):
    """绘制训练历史曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Training History', fontsize=16)

    metrics = ['loss', 'recall']

    for i, metric in enumerate(metrics):
        ax = axes[i]

        if metric in train_history and metric in val_history:
            epochs = range(1, len(train_history[metric]) + 1)
            ax.plot(epochs, train_history[metric], 'b-', label=f'Train {metric}')
            ax.plot(epochs, val_history[metric], 'r-', label=f'Val {metric}')
            ax.set_title(f'{metric.capitalize()} History')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"训练历史图已保存到: {save_path}")

    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: Path = None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Selected', 'Selected'],
                yticklabels=['Not Selected', 'Selected'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"混淆矩阵已保存到: {save_path}")

    plt.show()

def save_metrics_to_json(metrics: Dict, save_path: Path):
    """将指标保存为JSON文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logging.info(f"指标已保存到: {save_path}")

def set_random_seed(config, seed: int = None):
    """设置随机种子以确保结果可复现"""
    seed = seed or config['RANDOM_SEED']

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"随机种子已设置为: {seed}")

def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """统计模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params

def get_model_size(model: torch.nn.Module) -> float:
    """获取模型大小（MB）"""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def print_model_info(model: torch.nn.Module):
    """打印模型信息"""
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)

    logging.info("=" * 50)
    logging.info("模型信息:")
    logging.info(f"总参数数量: {total_params:,}")
    logging.info(f"可训练参数数量: {trainable_params:,}")
    logging.info(f"模型大小: {model_size:.2f} MB")
    logging.info("=" * 50)

# ==================== 配置和路径管理函数 ====================



# ==================== 模型创建函数 ====================

def create_model(config):
    """创建模型"""
    from model import QueryAwareGNN

    model = QueryAwareGNN(
        input_dim=config['EMBEDDING_DIM'],
        hidden_dim=config['HIDDEN_DIM'],
        gnn_layers=config['GNN_LAYERS'],
        mlp_hidden_dim=config['MLP_HIDDEN_DIM'],
        dropout_rate=config['DROPOUT_RATE'],
        gnn_type=config['GNN_TYPE']
    )
    return model

def print_model_info(model):
    """打印模型信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"模型参数统计:")
    logging.info(f"  总参数数: {total_params:,}")
    logging.info(f"  可训练参数数: {trainable_params:,}")
    logging.info(f"  模型大小: {total_params * 4 / 1024 / 1024:.2f} MB")

# ==================== 训练工具函数 ====================

def get_query_threshold_predictions(tool_probs, query_probs, batch):
    """
    使用query概率作为阈值进行预测：高于query节点概率的选，否则不选

    Args:
        tool_probs: 工具选择概率 [total_tool_nodes]
        query_probs: 查询概率 [batch_size]
        batch: PyG批次对象

    Returns:
        predictions: 二值预测结果 [total_tool_nodes]
    """
    import torch

    # 获取每个工具节点对应的图索引
    _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)

    # 为每个工具节点找到对应的查询概率
    tool_batch_indices = batch.batch[tool_mask]  # 工具节点对应的图索引
    tool_query_probs = query_probs[tool_batch_indices]  # 对应的查询概率

    # 直接使用query概率作为阈值：高于query节点概率的选，否则不选
    predictions = (tool_probs > tool_query_probs).float()

    return predictions

def log_sample_details(batch, tool_probs, predictions, labels, query_probs, logger, phase="训练", threshold_mode: str = "query"):
    """记录第一个样本的详细信息"""
    import torch

    try:
        # 获取第一个图的信息
        first_graph_mask = (batch.batch == 0)
        num_nodes_first_graph = first_graph_mask.sum().item()

        # 获取第一个图的查询文本
        if hasattr(batch, 'query_text') and len(batch.query_text) > 0:
            query_text = batch.query_text[0][:100] + "..." if len(batch.query_text[0]) > 100 else batch.query_text[0]
        else:
            query_text = "未知查询"

        # 计算第一个样本的边数（只统计第一个样本内的边）
        first_node_mask = (batch.batch == 0)
        edge_mask = first_node_mask[batch.edge_index[0]] & first_node_mask[batch.edge_index[1]]
        first_graph_edges = batch.edge_index[:, edge_mask]

        logger.info(f"{phase}样本详情 - 第一个样本")
        logger.info(f"Query: {query_text}")
        logger.info(f"图结构: {num_nodes_first_graph} 个节点 (1查询+{num_nodes_first_graph-1}工具), {first_graph_edges.size(1) // 2} 条边")

        # 获取第一个图的工具节点信息
        # 计算第一个图中工具节点的数量
        num_tools_first_graph = num_nodes_first_graph - 1  # 减去查询节点

        # 获取第一个图的工具节点数据（从工具概率/预测/标签的开头取对应数量）
        first_graph_tool_probs = tool_probs[:num_tools_first_graph]
        first_graph_predictions = predictions[:num_tools_first_graph]
        first_graph_labels = labels[:num_tools_first_graph]

        num_candidate_tools = len(first_graph_tool_probs)
        logger.info(f"候选工具数: {num_candidate_tools}")

        # 显示每个工具的详细信息
        score_label = "概率" if threshold_mode == 'fixed' else "分数"
        for i in range(min(num_candidate_tools, 10)):  # 最多显示10个工具
            score = first_graph_tool_probs[i].item()
            pred = "选中" if first_graph_predictions[i].item() > 0.5 else "未选"
            actual = "选中" if first_graph_labels[i].item() > 0.5 else "未选"
            correct = "✓" if first_graph_predictions[i].item() == first_graph_labels[i].item() else "✗"
            logger.info(f"  API{i+1}: {score_label}={score:.3f}, 预测={pred}, 实际={actual} {correct}")

        # 统计信息
        predicted_selected = (first_graph_predictions > 0.5).sum().item()
        actual_selected = (first_graph_labels > 0.5).sum().item()
        correct_predictions = (first_graph_predictions == first_graph_labels).sum().item()

        logger.info(f"预测选中: {predicted_selected}/{num_candidate_tools} 个API")
        logger.info(f"实际选中: {actual_selected}/{num_candidate_tools} 个API")
        logger.info(f"正确预测: {correct_predictions} 个API")

        # 显示当前阈值（根据模式）
        if threshold_mode == 'fixed':
            logger.info(f"当前阈值: 0.500 (固定)")
        else:
            if len(query_probs) > 0:
                current_threshold = query_probs[0].item()
                logger.info(f"当前阈值: {current_threshold:.3f} (动态)")

    except Exception as e:
        logger.warning(f"记录样本详情时出错: {e}")

def create_node_masks(batch_idx, num_graphs):
    """创建查询节点和工具节点的掩码"""
    import torch

    # 找到每个图的第一个节点（查询节点）
    query_mask = torch.zeros(len(batch_idx), dtype=torch.bool, device=batch_idx.device)

    # 为每个图找到第一个节点的位置
    for graph_id in range(num_graphs):
        graph_nodes = torch.where(batch_idx == graph_id)[0]
        if len(graph_nodes) > 0:
            query_mask[graph_nodes[0]] = True

    tool_mask = ~query_mask
    return query_mask, tool_mask

def extract_tool_labels(batch, device):
    """从批次中提取工具节点的标签"""
    import torch

    # PyG的DataLoader会自动将所有图的y标签连接成一个张量
    # 我们的y包含的是每个图中工具节点的标签（不包括查询节点）

    if hasattr(batch, 'y') and batch.y is not None:
        # 确保标签在正确的设备上
        labels = batch.y.to(device)

        # 确保标签是float类型
        if labels.dtype != torch.float:
            labels = labels.float()

        return labels
    else:
        # 如果没有标签，创建零标签
        # 计算工具节点总数（总节点数 - 查询节点数）
        total_tool_nodes = batch.x.size(0) - batch.num_graphs
        return torch.zeros(total_tool_nodes, dtype=torch.float, device=device)

def extract_tool_batch_indices(batch):
    """提取工具节点对应的批次索引"""
    # 创建工具节点掩码
    _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)
    # 返回工具节点对应的批次索引
    return batch.batch[tool_mask]

def make_predictions(tool_logits, query_scores, batch, threshold_mode='query'):
    """根据阈值模式进行预测
    - fixed: 对 tool_logits 做 sigmoid 后与 0.5 比较
    - query: 直接比较 logits：tool_logits > query_logits[tool_idx]
    """
    import torch

    if threshold_mode == 'fixed':
        tool_probs = torch.sigmoid(tool_logits)
        predictions = (tool_probs > 0.5).float()
    else:
        predictions = get_query_threshold_predictions_logits(tool_logits, query_scores, batch)

    return predictions


def get_query_threshold_predictions_logits(tool_logits, query_logits, batch):
    """
    使用query的logits作为阈值：tool_logits > 对应的query_logits 则选中
    返回float二值张量
    """
    import torch
    # 获取每个工具节点对应的图索引
    _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)
    tool_batch_indices = batch.batch[tool_mask]

    # 为每个工具节点找到对应的query logits
    tool_query_logits = query_logits[tool_batch_indices]

    # 直接比较logits
    predictions = (tool_logits > tool_query_logits).float()
    return predictions


# ==================== 训练辅助函数 ====================

def create_node_masks(batch_idx, num_graphs):
    """创建查询节点和工具节点的掩码"""
    import torch

    # 找到每个图的第一个节点（查询节点）
    query_mask = torch.zeros(len(batch_idx), dtype=torch.bool, device=batch_idx.device)

    # 为每个图找到第一个节点的位置
    for graph_id in range(num_graphs):
        graph_nodes = torch.where(batch_idx == graph_id)[0]
        if len(graph_nodes) > 0:
            query_mask[graph_nodes[0]] = True

    tool_mask = ~query_mask
    return query_mask, tool_mask

def extract_tool_labels(batch, device):
    """从批次中提取工具节点的标签"""
    import torch

    # PyG的DataLoader会自动将所有图的y标签连接成一个张量
    # 我们的y包含的是每个图中工具节点的标签（不包括查询节点）

    if hasattr(batch, 'y') and batch.y is not None:
        # 确保标签在正确的设备上
        labels = batch.y.to(device)

        # 确保标签是float类型
        if labels.dtype != torch.float:
            labels = labels.float()

        return labels
    else:
        # 如果没有标签，创建零标签
        # 计算工具节点总数（总节点数 - 查询节点数）
        total_tool_nodes = batch.x.size(0) - batch.num_graphs
        return torch.zeros(total_tool_nodes, dtype=torch.float, device=device)

def extract_tool_batch_indices(batch):
    """提取工具节点对应的批次索引"""
    # 创建工具节点掩码
    _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)
    # 返回工具节点对应的批次索引
    return batch.batch[tool_mask]

def make_predictions(tool_logits, query_scores, batch, threshold_mode='query'):
    """根据阈值模式进行预测
    - fixed: 对 tool_logits 做 sigmoid 后与 0.5 比较
    - query: 直接比较 logits：tool_logits > 对应的 query_logits
    """
    import torch

    if threshold_mode == 'fixed':
        tool_probs = torch.sigmoid(tool_logits)
        predictions = (tool_probs > 0.5).float()
    else:  # query模式
        predictions = get_query_threshold_predictions_logits(tool_logits, query_scores, batch)

    return predictions

def get_query_threshold_predictions(tool_probs, query_probs, batch):
    """
    使用query概率作为阈值进行预测：高于query节点概率的选，否则不选

    Args:
        tool_probs: 工具选择概率 [total_tool_nodes]
        query_probs: 查询概率 [batch_size]
        batch: PyG批次对象

    Returns:
        predictions: 二值预测结果 [total_tool_nodes]
    """
    # 为每个工具节点分配对应的query概率作为阈值
    # 需要根据batch.batch索引来映射

    # 获取每个工具节点对应的图索引
    _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)

    # 为每个工具节点找到对应的查询概率
    tool_batch_indices = batch.batch[tool_mask]  # 工具节点对应的图索引
    tool_query_probs = query_probs[tool_batch_indices]  # 对应的查询概率

    # 直接使用query概率作为阈值：高于query节点概率的选，否则不选
    predictions = (tool_probs > tool_query_probs).float()

    return predictions

