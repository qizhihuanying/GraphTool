"""
查询感知图神经网络模型
核心创新：将用户查询作为特殊节点集成到工具依赖图中
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import logging

logger = logging.getLogger(__name__)

class QueryAwareGNN(nn.Module):
    """查询感知的图神经网络模型"""
    
    def __init__(self, 
                 input_dim: int = None,
                 hidden_dim: int = None,
                 gnn_layers: int = None,
                 mlp_hidden_dim: int = None,
                 dropout_rate: float = None,
                 gnn_type: str = "GCN"):
        """
        初始化模型
        
        Args:
            input_dim: 输入嵌入维度
            hidden_dim: GNN隐藏层维度
            gnn_layers: GNN层数
            mlp_hidden_dim: 输出MLP隐藏层维度
            dropout_rate: Dropout率
            gnn_type: GNN类型 ("GCN" 或 "GAT")
        """
        super(QueryAwareGNN, self).__init__()
        
        # 设置模型参数（移除默认值依赖，要求明确传入）
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gnn_layers = gnn_layers
        self.mlp_hidden_dim = mlp_hidden_dim
        self.dropout_rate = dropout_rate
        self.gnn_type = gnn_type
        
        logger.info(f"初始化QueryAwareGNN: {self.gnn_type}, "
                   f"输入维度={self.input_dim}, 隐藏维度={self.hidden_dim}, "
                   f"GNN层数={self.gnn_layers}")
        
        # 1. 统一嵌入对齐层：所有节点共享一套线性层
        self.embedding_aligner = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 2. GNN层
        self.gnn_layers_list = nn.ModuleList()
        
        if self.gnn_type == "GCN":
            # 使用图卷积网络
            for i in range(self.gnn_layers):
                self.gnn_layers_list.append(GCNConv(self.hidden_dim, self.hidden_dim))
        elif self.gnn_type == "GAT":
            # 使用图注意力网络
            for i in range(self.gnn_layers):
                self.gnn_layers_list.append(
                    GATConv(self.hidden_dim, self.hidden_dim, heads=1, dropout=self.dropout_rate)
                )
        else:
            raise ValueError(f"不支持的GNN类型: {self.gnn_type}")
        
        # 3. Dropout层
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 4. 共享输出MLP：工具和查询节点使用同一个MLP，确保在同一概率空间
        self.output_mlp = nn.Sequential(
            # nn.Linear(self.hidden_dim, self.mlp_hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(self.dropout_rate),
            # nn.Linear(self.mlp_hidden_dim, 1)
            nn.Linear(self.hidden_dim, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, batch):
        """
        前向传播
        
        Args:
            batch: PyG Batch对象，包含批处理的图数据
            
        Returns:
            tool_probs: 工具选择概率 [batch_size * num_tools_per_sample]
            query_probs: 查询节点输出概率（可选）
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        # 1. 统一嵌入对齐：残差连接保留原始语义
        x = x + self.embedding_aligner(x)

        # 2. GNN消息传递
        for i, gnn_layer in enumerate(self.gnn_layers_list):
            h = gnn_layer(x, edge_index)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h
        
        # 4. 分离查询节点和工具节点
        # 在我们的设计中，每个子图的节点0是查询节点，其余是工具节点
        query_mask, tool_mask = self._create_node_masks(batch_idx, batch.num_graphs)

        query_embeddings = x[query_mask]  # [batch_size, hidden_dim]
        tool_embeddings = x[tool_mask]    # [total_tool_nodes, hidden_dim]

        # 5. 生成输出 logits
        tool_logits = self.output_mlp(tool_embeddings).squeeze(-1)   # [total_tool_nodes]
        query_logits = self.output_mlp(query_embeddings).squeeze(-1) # [batch_size]

        return tool_logits, query_logits
    
    def _create_node_masks(self, batch_idx, num_graphs):
        """
        创建查询节点和工具节点的掩码
        
        Args:
            batch_idx: 节点的批次索引
            num_graphs: 批次中图的数量
            
        Returns:
            query_mask: 查询节点掩码
            tool_mask: 工具节点掩码
        """
        # 找到每个图的第一个节点（查询节点）
        query_mask = torch.zeros(len(batch_idx), dtype=torch.bool, device=batch_idx.device)
        
        # 为每个图找到第一个节点的位置
        for graph_id in range(num_graphs):
            graph_nodes = (batch_idx == graph_id).nonzero(as_tuple=True)[0]
            if len(graph_nodes) > 0:
                first_node_idx = graph_nodes[0]
                query_mask[first_node_idx] = True
        
        tool_mask = ~query_mask
        
        return query_mask, tool_mask
    
    def predict(self, batch, threshold: float = 0.5, use_dynamic_threshold: bool = False):
        """
        预测工具选择
        
        Args:
            batch: 输入批次
            threshold: 固定阈值
            use_dynamic_threshold: 是否使用查询节点输出作为动态阈值
            
        Returns:
            predictions: 二元预测结果
        """
        self.eval()
        with torch.no_grad():
            tool_logits, query_probs = self.forward(batch)
            # 将logits转换为概率用于预测
            tool_probs = torch.sigmoid(tool_logits)

            if use_dynamic_threshold:
                # 使用查询节点输出作为动态阈值
                # 需要将查询概率扩展到对应的工具节点
                batch_idx = batch.batch
                _, tool_mask = self._create_node_masks(batch_idx, batch.num_graphs)

                # 为每个工具节点分配对应的查询阈值
                tool_batch_idx = batch_idx[tool_mask]
                dynamic_thresholds = query_probs[tool_batch_idx]
                predictions = (tool_probs > dynamic_thresholds).float()
            else:
                # 使用更低的固定阈值以提高召回率
                predictions = (tool_probs > threshold).float()
            
            return predictions


