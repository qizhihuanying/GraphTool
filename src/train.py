"""
训练模块：实现模型训练和评估逻辑
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List
import os

from model import QueryAwareGNN
from sklearn.metrics import f1_score
from utils import (calculate_metrics, EarlyStopping, extract_tool_labels,
                   make_predictions, log_sample_details)

logger = logging.getLogger(__name__)

class Trainer:
    """模型训练器"""
    
    def __init__(self,
                 model: QueryAwareGNN,
                 train_loader,
                 val_loader,
                 test_loader=None,
                 config=None,
                 device: str = None,
                 logger=None,
                 threshold_mode: str = 'query'):
        """
        初始化训练器

        Args:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            test_loader: 测试数据加载器（可选）
            config: 配置字典
            device: 训练设备
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device or config['DEVICE']
        self.logger = logger or logging.getLogger(__name__)
        self.threshold_mode = threshold_mode

        # 为当前实验生成唯一的最佳模型路径（使用logger名称保证与日志名一致）
        models_dir = self.config['MODELS_DIR']
        run_name = self.logger.name if self.logger else 'run'
        self.best_model_path = models_dir / f"{run_name}_best.pt"

        # 将模型移到指定设备
        self.model.to(self.device)
        
        # 设置优化器
        if config['OPTIMIZER'] == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config['LEARNING_RATE'],
                weight_decay=config['WEIGHT_DECAY']
            )
        else:
            raise ValueError(f"不支持的优化器: {config['OPTIMIZER']}")

        # 设置损失函数（使用带logits的稳定版本）
        self.criterion = nn.BCEWithLogitsLoss()

        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config['EARLY_STOPPING_PATIENCE'],
            min_delta=1e-4
        )
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'recall': []
        }

        self.val_history = {
            'loss': [],
            'recall': []
        }
        

    
    def train_epoch(self) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(self.train_loader, desc="训练中")

        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            batch = batch.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            tool_logits, query_logits = self.model(batch)

            # 计算基于delta的损失：delta = tool - query
            labels = extract_tool_labels(batch, self.device)

            # 获取每个工具节点对应的查询节点logits
            from utils import create_node_masks
            _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)
            tool_batch_indices = batch.batch[tool_mask]  # 工具节点对应的图索引
            corresponding_query_logits = query_logits[tool_batch_indices]  # 对应的查询logits

            # 计算delta = tool_logits - query_logits
            delta_logits = tool_logits - corresponding_query_logits

            # 对delta应用BCEWithLogitsLoss，让正样本delta>0，负样本delta<0
            loss = self.criterion(delta_logits, labels)

            # 反向传播
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录统计信息
            total_loss += loss.item()

            # 收集预测和标签用于指标计算（query阈值使用logits对比）
            predictions = make_predictions(tool_logits, query_logits, batch, self.threshold_mode)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 详细日志：展示第一个batch的第一个样本
            if batch_idx == 0:
                # 为日志准备可读分数：fixed模式下用概率，query模式下直接用logits
                if self.threshold_mode == 'fixed':
                    tool_scores_for_log = torch.sigmoid(tool_logits)
                    query_scores_for_log = torch.sigmoid(query_logits)
                else:
                    tool_scores_for_log = tool_logits
                    query_scores_for_log = query_logits
                log_sample_details(batch, tool_scores_for_log, predictions, labels, query_scores_for_log, self.logger, "训练", self.threshold_mode)

            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # 计算epoch指标
        avg_loss = total_loss / len(self.train_loader)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))

        # 计算F1分数
        f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='binary', zero_division=0)

        epoch_metrics = {
            'loss': avg_loss,
            'recall': metrics['recall'],
            'f1': f1
        }
        
        return epoch_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """在验证集上评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            # 验证进度条始终显示在终端
            progress_bar = tqdm(self.val_loader, desc="验证中", disable=False)

            for batch_idx, batch in enumerate(progress_bar):
                batch = batch.to(self.device)
                
                # 前向传播
                tool_logits, query_logits = self.model(batch)

                # 计算损失：按阈值模式对齐训练目标
                labels = extract_tool_labels(batch, self.device)
                # 基于delta的损失：delta = tool - query
                from utils import create_node_masks
                _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)
                tool_batch_indices = batch.batch[tool_mask]
                corresponding_query_logits = query_logits[tool_batch_indices]
                delta_logits = tool_logits - corresponding_query_logits
                loss = self.criterion(delta_logits, labels)

                total_loss += loss.item()

                # 收集预测和标签（query阈值使用logits对比）
                predictions = make_predictions(tool_logits, query_logits, batch, self.threshold_mode)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 详细日志：展示第一个batch的第一个样本
                if batch_idx == 0:
                    if self.threshold_mode == 'fixed':
                        tool_scores_for_log = torch.sigmoid(tool_logits)
                        query_scores_for_log = torch.sigmoid(query_logits)
                    else:
                        tool_scores_for_log = tool_logits
                        query_scores_for_log = query_logits
                    log_sample_details(batch, tool_scores_for_log, predictions, labels, query_scores_for_log, self.logger, "验证", self.threshold_mode)

                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # 计算指标
        avg_loss = total_loss / len(self.val_loader)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))

        # 计算F1分数
        f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='binary', zero_division=0)

        eval_metrics = {
            'loss': avg_loss,
            'recall': metrics['recall'],
            'f1': f1
        }
        
        return eval_metrics

    def test(self) -> Dict[str, float]:
        """在测试集上评估模型"""
        if self.test_loader is None:
            return {}
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            # 测试进度条始终显示在终端
            progress_bar = tqdm(self.test_loader, desc="测试中", disable=False)

            for batch in progress_bar:
                batch = batch.to(self.device)

                # 前向传播
                tool_logits, query_logits = self.model(batch)

                # 计算损失
                labels = extract_tool_labels(batch, self.device)
                # 基于delta的损失：delta = tool - query
                from utils import create_node_masks
                _, tool_mask = create_node_masks(batch.batch, batch.num_graphs)
                tool_batch_indices = batch.batch[tool_mask]
                corresponding_query_logits = query_logits[tool_batch_indices]
                delta_logits = tool_logits - corresponding_query_logits
                loss = self.criterion(delta_logits, labels)

                # 收集预测和标签（query阈值使用logits对比）
                predictions = make_predictions(tool_logits, query_logits, batch, self.threshold_mode)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                total_loss += loss.item()

                # 更新进度条
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        # 计算指标
        avg_loss = total_loss / len(self.test_loader)
        metrics = calculate_metrics(np.array(all_labels), np.array(all_predictions))

        # 计算F1分数
        f1 = f1_score(np.array(all_labels), np.array(all_predictions), average='binary', zero_division=0)

        test_metrics = {
            'loss': avg_loss,
            'recall': metrics['recall'],
            'f1': f1
        }



        return test_metrics




    def train(self, num_epochs: int = None) -> Dict[str, List[float]]:
        """完整的训练循环"""
        num_epochs = num_epochs or self.config['NUM_EPOCHS']

        best_val_loss = float('inf')
        best_model_path = self.best_model_path

        for epoch in range(num_epochs):
            
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.evaluate()
            
            # 记录历史
            for key in self.train_history.keys():
                self.train_history[key].append(train_metrics[key])
                self.val_history[key].append(val_metrics[key])
            
            # 打印指标
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Epoch: {epoch+1}, Train Loss: {train_metrics['loss']:.4f}, "
                       f"Val Loss: {val_metrics['loss']:.4f}, Recall: {val_metrics['recall']:.4f}, "
                       f"F1: {val_metrics['f1']:.4f}")
            self.logger.info(f"{'='*60}")

            # 保存最佳模型
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_model(best_model_path)

            # 早停检查
            if self.early_stopping(val_metrics['loss']):
                break

        # 训练完成后在测试集上评估
        test_metrics = {}
        if self.test_loader is not None:
            # 加载最佳模型进行测试
            if best_model_path.exists():
                try:
                    self.load_model(best_model_path)
                except Exception as e:
                    self.logger.warning(f"加载最佳模型失败（可能是结构不一致），跳过使用当前模型测试。原因: {e}")

            test_metrics = self.test()

            self.logger.info(f"{'='*60}")
            self.logger.info(f"Test Loss: {test_metrics['loss']:.4f}, Test Recall: {test_metrics['recall']:.4f}, "
                       f"Test F1: {test_metrics['f1']:.4f}")
            self.logger.info(f"{'='*60}")

        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'test_metrics': test_metrics
        }
    
    def save_model(self, path: Path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history
        }, path)
    
    def load_model(self, path: Path):
        """加载模型"""
        # 为了兼容PyTorch 2.6的安全限制，设置weights_only=False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        if 'val_history' in checkpoint:
            self.val_history = checkpoint['val_history']


