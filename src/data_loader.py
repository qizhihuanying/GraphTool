import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import logging
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 设置日志
logger = logging.getLogger(__name__)

class EmbeddingGenerator:

    def __init__(self, config, model_name: str = None):
        self.config = config
        self.model_name = model_name or config['EMBEDDING_MODEL_NAME']
        self.tokenizer = None
        self.model = None
        self.device = config['DEVICE']
        self._embedding_cache = {}
        self._load_model()

    def _load_model(self):
        try:
            # 设备校验与自动降级
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning("CUDA 不可用，自动降级为 CPU 运行")
                self.device = 'cpu'
                self.config['DEVICE'] = 'cpu'
            elif self.device.startswith('cuda:'):
                # 检查指定的GPU是否存在
                gpu_id = int(self.device.split(':')[1])
                if gpu_id >= torch.cuda.device_count():
                    logger.warning(f"指定的GPU {gpu_id} 不存在，自动降级为默认CUDA设备")
                    self.device = 'cuda'
                    self.config['DEVICE'] = 'cuda'

            logger.info(f"正在加载Qwen3-Embedding模型: {self.model_name}")
            logger.info(f"使用设备: {self.device}")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

            # 按配置选择精度加载模型
            dtype = self.config.get('EMBED_DTYPE', 'fp16')
            torch_dtype = torch.float16 if dtype == 'fp16' else torch.float32
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch_dtype
            )

            self.model = self.model.to(self.device)
            self.model.eval()

            # 尝试从配置中探测隐藏维度并对齐 EMBEDDING_DIM
            detected_dim = None
            if hasattr(self.model, 'config'):
                detected_dim = getattr(self.model.config, 'hidden_size', None)
                if detected_dim is None:
                    detected_dim = getattr(self.model.config, 'projection_dim', None)

            if isinstance(detected_dim, int) and detected_dim > 0:
                if self.config.get('EMBEDDING_DIM') != detected_dim:
                    logger.warning(f"自动校准 EMBEDDING_DIM: {self.config.get('EMBEDDING_DIM')} -> {detected_dim}")
                    self.config['EMBEDDING_DIM'] = detected_dim

            logger.info("Qwen3-Embedding模型加载成功")
            logger.info(f"模型配置 - 嵌入维度: {self.config['EMBEDDING_DIM']}, 最大序列长度: {self.config['MAX_SEQUENCE_LENGTH']}")
            logger.info(f"模型设备: {next(self.model.parameters()).device}")
        except Exception as e:
            # 明确失败：后续不再尝试回退到 sentence-transformers
            logger.error(f"加载Qwen3-Embedding模型失败: {e}")
            raise

    def generate_embedding(self, text: str) -> torch.Tensor:
        """生成单个文本的嵌入向量，带缓存机制"""
        if not text or not text.strip():
            return torch.zeros(self.config['EMBEDDING_DIM'])

        # 检查缓存
        text_key = text.strip()
        if text_key in self._embedding_cache:
            return self._embedding_cache[text_key].clone()

        try:
            # 严格 HF 分支：若模型/分词器不存在，立即抛错
            if self.tokenizer is None or self.model is None:
                raise RuntimeError("Qwen 模型未正确加载，无法生成嵌入")

            inputs = self.tokenizer(
                text,
                max_length=self.config['MAX_SEQUENCE_LENGTH'],
                truncation=True,
                padding=True,
                return_tensors="pt"
            )

            # 将输入移动到模型设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                last_hidden = outputs.last_hidden_state  # [B, L, H]
                attn = inputs.get('attention_mask', None)
                if attn is not None:
                    attn = attn.unsqueeze(-1).type_as(last_hidden)  # [B, L, 1]
                    sum_emb = (last_hidden * attn).sum(dim=1)
                    denom = attn.sum(dim=1).clamp_min(1e-6)
                    embedding = (sum_emb / denom).squeeze(0)
                else:
                    embedding = last_hidden.mean(dim=1).squeeze(0)

            embedding = embedding.cpu()  # 返回CPU上的张量

            # 缓存嵌入
            self._embedding_cache[text_key] = embedding.clone()
            return embedding

        except Exception as e:
            logger.warning(f"生成嵌入失败，返回零向量: {e}")
            embedding = torch.zeros(self.config['EMBEDDING_DIM'])
            self._embedding_cache[text_key] = embedding.clone()
            return embedding

    def generate_batch_embeddings(self, texts: List[str]) -> torch.Tensor:
        """批量生成嵌入向量（带进度条和缓存）"""
        try:
            # 过滤空文本
            valid_texts = [text if text and text.strip() else "empty text" for text in texts]

            # 检查缓存，分离已缓存和未缓存的文本
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(valid_texts):
                text_key = text.strip()
                if text_key in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[text_key].clone()))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            logger.info(f"缓存命中: {len(cached_embeddings)}/{len(valid_texts)}, 需要生成: {len(uncached_texts)}")

            if self.tokenizer is not None and uncached_texts:
                # 使用Qwen模型批量处理未缓存的文本
                new_embeddings = []
                batch_size = int(self.config.get('EMBED_BATCH_SIZE', 8))  # 控制批大小避免内存溢出

                # 添加进度条
                from tqdm import tqdm
                total_batches = (len(uncached_texts) + batch_size - 1) // batch_size
                progress_bar = tqdm(
                    range(0, len(uncached_texts), batch_size),
                    desc="生成嵌入向量",
                    unit="批次",
                    total=total_batches
                )

                for i in progress_bar:
                    batch_texts = uncached_texts[i:i+batch_size]

                    inputs = self.tokenizer(
                        batch_texts,
                        max_length=self.config['MAX_SEQUENCE_LENGTH'],
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    )

                    # 将输入移动到模型设备
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        last_hidden = outputs.last_hidden_state  # [B, L, H]
                        attn = inputs.get('attention_mask', None)
                        if attn is not None:
                            attn = attn.unsqueeze(-1).type_as(last_hidden)  # [B, L, 1]
                            sum_emb = (last_hidden * attn).sum(dim=1)
                            denom = attn.sum(dim=1).clamp_min(1e-6)
                            batch_embeddings = (sum_emb / denom)
                        else:
                            batch_embeddings = last_hidden.mean(dim=1)
                        batch_embeddings_cpu = batch_embeddings.cpu()  # 移回CPU
                        new_embeddings.append(batch_embeddings_cpu)

                        # 缓存新生成的嵌入
                        for j, text in enumerate(batch_texts):
                            text_key = text.strip()
                            self._embedding_cache[text_key] = batch_embeddings_cpu[j].clone()

                    # 更新进度条信息
                    progress_bar.set_postfix({
                        '已处理': min(i + batch_size, len(uncached_texts)),
                        '总数': len(uncached_texts),
                        '维度': self.config['EMBEDDING_DIM']
                    })

                progress_bar.close()

                # 合并缓存的和新生成的嵌入
                if new_embeddings:
                    new_embeddings_tensor = torch.cat(new_embeddings, dim=0)
                else:
                    new_embeddings_tensor = torch.empty(0, self.config['EMBEDDING_DIM'])

                # 重新排列到原始顺序
                all_embeddings = torch.zeros(len(valid_texts), self.config['EMBEDDING_DIM'])

                # 填入缓存的嵌入
                for idx, embedding in cached_embeddings:
                    all_embeddings[idx] = embedding

                # 填入新生成的嵌入
                for i, idx in enumerate(uncached_indices):
                    if i < new_embeddings_tensor.size(0):
                        all_embeddings[idx] = new_embeddings_tensor[i]

                return all_embeddings
            elif uncached_texts:
                # 严格 HF 分支：若模型/分词器不存在，立即抛错
                raise RuntimeError("Qwen 模型未正确加载，无法批量生成嵌入")
            else:
                # 所有文本都已缓存
                all_embeddings = torch.zeros(len(valid_texts), self.config['EMBEDDING_DIM'])
                for idx, embedding in cached_embeddings:
                    all_embeddings[idx] = embedding
                return all_embeddings

        except Exception as e:
            logger.warning(f"批量生成嵌入失败，使用单个生成: {e}")
            # 回退到单个生成（带进度条）
            from tqdm import tqdm
            embeddings = []

            progress_bar = tqdm(texts, desc="单个生成嵌入", unit="文本")
            for text in progress_bar:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)

                progress_bar.set_postfix({
                    '已完成': len(embeddings),
                    '维度': self.config['EMBEDDING_DIM']
                })

            progress_bar.close()
            return torch.stack(embeddings)

    def save_cache(self, cache_path: Path):
        """保存嵌入缓存到文件"""
        try:
            torch.save(self._embedding_cache, cache_path)
            logger.info(f"嵌入缓存已保存到: {cache_path} (共 {len(self._embedding_cache)} 个条目)")
        except Exception as e:
            logger.warning(f"保存嵌入缓存失败: {e}")

    def load_cache(self, cache_path: Path):
        """从文件加载嵌入缓存"""
        try:
            if cache_path.exists():
                self._embedding_cache = torch.load(cache_path, map_location='cpu')
                logger.info(f"嵌入缓存已加载: {cache_path} (共 {len(self._embedding_cache)} 个条目)")
            else:
                logger.info(f"缓存文件不存在: {cache_path}")
        except Exception as e:
            logger.warning(f"加载嵌入缓存失败: {e}")
            self._embedding_cache = {}

    def clear_cache(self):
        """清空嵌入缓存"""
        self._embedding_cache.clear()
        logger.info("嵌入缓存已清空")

def load_toolbench_g3_data(data_path: Path) -> List[Dict]:
    """加载ToolBench G3数据集"""
    logger.info(f"正在加载ToolBench G3数据: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    try:
        # ToolBench数据是标准JSON数组格式
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"成功加载 {len(data)} 条ToolBench G3数据")
        return data
    except Exception as e:
        logger.error(f"加载ToolBench G3数据失败: {e}")
        raise


def build_tool_graph_toolbench(config, all_data: List[Dict], train_val_data: List[Dict], embedding_generator: EmbeddingGenerator) -> Tuple[torch.Tensor, Dict[str, int], torch.Tensor]:
    """
    基于ToolBench G3数据构建工具依赖图

    Args:
        all_data: 所有数据（用于收集工具节点，避免test中的新工具无法处理）
        train_val_data: 训练+验证数据（用于计算边的共现，避免数据泄露）
        embedding_generator: 嵌入生成器

    Returns:
        node_features: 工具节点特征矩阵 [num_tools, embedding_dim]
        tool_to_idx: 工具名到索引的映射
        edge_index: 边索引 [2, num_edges]
    """
    logger.info("开始构建ToolBench工具依赖图...")

    # 1. 收集所有工具和文档
    tool_docs = {}
    candidate_tool_sets = []

    # 从所有数据中收集工具（确保测试集中的工具也被包含）
    for sample in all_data:
        if 'api_list' not in sample:
            continue

        candidate_tools = []
        for api in sample['api_list']:
            # 使用 tool_name::api_name 作为节点标识符
            tool_id = f"{api['tool_name']}::{api['api_name']}"
            candidate_tools.append(tool_id)

            # 构建工具文档
            if tool_id not in tool_docs:
                # Compose a rich tool description in English
                doc_parts = [
                    f"Tool: {api['tool_name']}",
                    f"API: {api['api_name']}",
                    f"Category: {api['category_name']}",
                    f"Description: {api['api_description']}",
                    f"Method: {api['method']}"
                ]

                # Append parameter info
                if api.get('required_parameters'):
                    required_params = [p['name'] for p in api['required_parameters']]
                    doc_parts.append(f"Required params: {', '.join(required_params)}")

                if api.get('optional_parameters'):
                    optional_params = [p['name'] for p in api['optional_parameters']]
                    doc_parts.append(f"Optional params: {', '.join(optional_params)}")

                tool_docs[tool_id] = " | ".join(doc_parts)

        if candidate_tools:
            candidate_tool_sets.append(candidate_tools)

    # 2. 统计工具共现关系（只使用训练+验证数据）
    tool_cooccurrence = defaultdict(lambda: defaultdict(int))

    for sample in train_val_data:
        if 'relevant APIs' not in sample:
            continue

        # 提取选中的工具
        selected_tools = []
        for tool_api_pair in sample['relevant APIs']:
            try:
                if isinstance(tool_api_pair, list) and len(tool_api_pair) >= 2:
                    tool_name, api_name = tool_api_pair[0], tool_api_pair[1]
                    tool_id = f"{tool_name}::{api_name}"
                    selected_tools.append(tool_id)
                else:
                    logger.warning(f"跳过格式异常的relevant API: {tool_api_pair}")
            except Exception as e:
                logger.warning(f"解析relevant API时出错: {tool_api_pair}, 错误: {e}")
                continue

        # 统计共现
        for i, tool1 in enumerate(selected_tools):
            for j, tool2 in enumerate(selected_tools):
                if i != j and tool1 in tool_docs and tool2 in tool_docs:
                    tool_cooccurrence[tool1][tool2] += 1

    # 过滤掉空的工具名称
    tool_docs = {k: v for k, v in tool_docs.items() if k}

    # 创建工具到索引的映射
    tools = list(tool_docs.keys())
    tool_to_idx = {tool: idx for idx, tool in enumerate(tools)}

    logger.info(f"发现 {len(tools)} 个唯一API工具")

    if not tools:
        logger.error("没有找到任何有效的API工具！")
        return torch.empty(0, config['EMBEDDING_DIM']), {}, torch.empty((2, 0), dtype=torch.long)

    # 生成工具嵌入
    logger.info(f"正在生成 {len(tools)} 个API工具的嵌入向量...")
    tool_texts = [tool_docs[tool] for tool in tools]
    node_features = embedding_generator.generate_batch_embeddings(tool_texts)
    logger.info(f"工具嵌入生成完成，形状: {node_features.shape}")

    # 构建无向边（基于候选工具共现，避免重复）
    edges = []
    edge_count = 0
    processed_pairs = set()  # 记录已处理的工具对

    for tool1, cooccur_dict in tool_cooccurrence.items():
        if tool1 in tool_to_idx:
            idx1 = tool_to_idx[tool1]
            for tool2, count in cooccur_dict.items():
                if (tool2 in tool_to_idx and
                    count >= config['CO_OCCURRENCE_THRESHOLD'] and
                    idx1 != tool_to_idx[tool2]):
                    idx2 = tool_to_idx[tool2]

                    # 确保无向边不重复（较小索引在前）
                    edge_pair = (min(idx1, idx2), max(idx1, idx2))
                    if edge_pair not in processed_pairs:
                        edges.append([idx1, idx2])
                        edges.append([idx2, idx1])  # 添加反向边
                        processed_pairs.add(edge_pair)
                        edge_count += 2

    # 转换为PyTorch张量
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    logger.info(f"ToolBench工具图构建完成: {len(tools)} 个节点, {edge_index.size(1)} 条边")
    logger.info(f"平均每个样本候选工具数: {sum(len(s) for s in candidate_tool_sets) / len(candidate_tool_sets):.2f}")

    return node_features, tool_to_idx, edge_index







def create_training_samples_toolbench(data: List[Dict], tool_to_idx: Dict[str, int], embedding_generator: EmbeddingGenerator) -> List[Dict]:
    """
    基于ToolBench G3数据创建样本

    Returns:
        samples: 包含查询嵌入、子图节点索引、标签的样本列表
    """
    logger.info("正在创建ToolBench样本...")
    samples = []
    skipped_no_candidates = 0
    skipped_no_valid_tools = 0
    skipped_no_labels = 0

    # 添加进度条
    from tqdm import tqdm
    progress_bar = tqdm(data, desc="创建样本", unit="样本")

    for sample in progress_bar:
        try:
            # 1. 提取用户查询
            if 'query' not in sample:
                continue
            user_query = sample['query']

            # 2. 提取候选工具
            if 'api_list' not in sample:
                skipped_no_candidates += 1
                continue

            candidate_tools = []
            for api in sample['api_list']:
                tool_id = f"{api['tool_name']}::{api['api_name']}"
                candidate_tools.append(tool_id)

            if not candidate_tools:
                skipped_no_candidates += 1
                continue

            # 3. 过滤出在工具图中存在的候选工具
            valid_candidate_tools = [tool for tool in candidate_tools if tool in tool_to_idx]

            if not valid_candidate_tools:
                skipped_no_valid_tools += 1
                continue

            # 4. 提取相关工具（标签）
            if 'relevant APIs' not in sample:
                skipped_no_labels += 1
                continue

            selected_tools = []
            for tool_api_pair in sample['relevant APIs']:
                try:
                    if isinstance(tool_api_pair, list) and len(tool_api_pair) >= 2:
                        tool_name, api_name = tool_api_pair[0], tool_api_pair[1]
                        tool_id = f"{tool_name}::{api_name}"
                        selected_tools.append(tool_id)
                    else:
                        logger.warning(f"跳过格式异常的relevant API: {tool_api_pair}")
                except Exception as e:
                    logger.warning(f"解析relevant API时出错: {tool_api_pair}, 错误: {e}")
                    continue

            if not selected_tools:
                skipped_no_labels += 1
                continue

            # 5. 生成查询嵌入
            query_embedding = embedding_generator.generate_embedding(user_query)

            # 6. 创建子图节点索引
            subgraph_node_indices = torch.tensor([tool_to_idx[tool] for tool in valid_candidate_tools], dtype=torch.long)

            # 7. 创建标签（多标签二元分类）
            labels = torch.zeros(len(valid_candidate_tools), dtype=torch.float)
            for i, tool in enumerate(valid_candidate_tools):
                if tool in selected_tools:
                    labels[i] = 1.0

            # 8. 检查是否有正标签
            if labels.sum() == 0:
                skipped_no_labels += 1
                continue

            # 9. 创建样本
            sample_data = {
                'query_embedding': query_embedding,
                'subgraph_node_indices': subgraph_node_indices,
                'label': labels,
                'query_text': user_query,  # 保存原始查询文本用于调试
                'candidate_tools': valid_candidate_tools,  # 保存候选工具列表用于调试
                'selected_tools': selected_tools  # 保存选中工具列表用于调试
            }

            samples.append(sample_data)

        except Exception as e:
            logger.warning(f"处理样本时出错: {e}")
            continue

    progress_bar.close()

    logger.info(f"ToolBench样本创建完成:")
    logger.info(f"  成功创建: {len(samples)} 个样本")
    logger.info(f"  跳过（无候选工具）: {skipped_no_candidates} 个")
    logger.info(f"  跳过（无有效工具）: {skipped_no_valid_tools} 个")
    logger.info(f"  跳过（无标签）: {skipped_no_labels} 个")

    if samples:
        # 统计信息
        avg_candidates = sum(len(s['candidate_tools']) for s in samples) / len(samples)
        avg_selected = sum(s['label'].sum().item() for s in samples) / len(samples)
        logger.info(f"  平均候选工具数: {avg_candidates:.2f}")
        logger.info(f"  平均选中工具数: {avg_selected:.2f}")

    return samples



def prepare_data(config):
    """准备ToolBench数据：自动合并多个指定/默认路径的数据（支持G2+G3），构建图与样本"""
    logger.info("开始ToolBench数据预处理（自动合并 G2+G3）...")

    # 检查是否已有预处理数据
    if (config['FULL_GRAPH_PATH'].exists() and
        config['TRAINING_SAMPLES_PATH'].exists() and
        config['VALIDATION_SAMPLES_PATH'].exists() and
        config['TEST_SAMPLES_PATH'].exists()):
        logger.info("发现已有预处理数据，跳过预处理步骤")
        return

    # 收集候选数据路径：优先使用 --dataset-path；若为空则默认 datasets/ToolBench 下的 G3 和 G2
    paths = []
    ds_path = config.get('dataset_path')
    if ds_path is not None:
        # 若传入的是目录，则合并目录下所有 *_query.json
        p = Path(ds_path)
        if p.is_dir():
            paths.extend(sorted(p.glob("*query.json")))
        else:
            paths.append(p)
    else:
        default_g3 = config['TOOLBENCH_DIR'] / 'G3_query.json'
        default_g2 = config['TOOLBENCH_DIR'] / 'G2_query.json'
        for p in [default_g3, default_g2]:
            if p.exists():
                paths.append(p)

    if not paths:
        raise FileNotFoundError("未找到任何 ToolBench 数据文件，请提供 --dataset-path 或将 G2/G3 放到 datasets/ToolBench/")

    # 加载并合并
    all_data = []
    total_each = []
    for p in paths:
        data = load_toolbench_g3_data(p)
        all_data.extend(data)
        total_each.append((p.name, len(data)))
    logger.info("已合并数据文件：" + ", ".join([f"{n}:{c}" for n,c in total_each]) + f"；合计 {len(all_data)} 条")

    # 随机打乱数据
    import random
    random.shuffle(all_data)
    logger.info("数据已随机打乱")

    # 划分训练、验证、测试数据
    from sklearn.model_selection import train_test_split

    train_val_data, test_data = train_test_split(
        all_data,
        test_size=config['TEST_SPLIT'],
        random_state=None
    )

    train_data, val_data = train_test_split(
        train_val_data,
        test_size=config['VAL_SPLIT'] / (config['TRAIN_SPLIT'] + config['VAL_SPLIT']),
        random_state=None
    )

    logger.info(f"数据划分完成:")
    logger.info(f"  训练集: {len(train_data)} 条样本 ({len(train_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  验证集: {len(val_data)} 条样本 ({len(val_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  测试集: {len(test_data)} 条样本 ({len(test_data)/len(all_data)*100:.1f}%)")

    # 初始化嵌入生成器
    logger.info("初始化嵌入生成器...")
    embedding_generator = EmbeddingGenerator(config)

    # 加载嵌入缓存
    cache_path = config['PREPROCESSED_DATA_DIR'] / "embedding_cache.pt"
    embedding_generator.load_cache(cache_path)

    # 构建ToolBench工具图（节点用所有数据，边用train+val数据）
    logger.info("构建ToolBench工具依赖图...")
    train_val_combined = train_data + val_data
    node_features, tool_to_idx, edge_index = build_tool_graph_toolbench(config, all_data, train_val_combined, embedding_generator)

    if len(tool_to_idx) == 0:
        raise ValueError("没有找到任何有效的API工具，请检查数据格式")

    # 保存完整图
    full_graph_data = {
        'node_features': node_features,
        'tool_to_idx': tool_to_idx,
        'edge_index': edge_index
    }
    torch.save(full_graph_data, config['FULL_GRAPH_PATH'])
    logger.info(f"ToolBench工具图已保存到: {config['FULL_GRAPH_PATH']}")

    # 分别为训练、验证、测试集创建样本
    logger.info("创建训练样本...")
    train_samples = create_training_samples_toolbench(train_data, tool_to_idx, embedding_generator)

    if len(train_samples) == 0:
        raise ValueError("没有创建任何有效的训练样本，请检查数据处理逻辑")

    logger.info("创建验证样本...")
    val_samples = create_training_samples_toolbench(val_data, tool_to_idx, embedding_generator)

    if len(val_samples) == 0:
        logger.warning("没有创建任何有效的验证样本")

    logger.info("创建测试样本...")
    test_samples = create_training_samples_toolbench(test_data, tool_to_idx, embedding_generator)

    if len(test_samples) == 0:
        logger.warning("没有创建任何有效的测试样本")

    # 保存样本
    torch.save(train_samples, config['TRAINING_SAMPLES_PATH'])
    torch.save(val_samples, config['VALIDATION_SAMPLES_PATH'])
    torch.save(test_samples, config['TEST_SAMPLES_PATH'])

    # 保存嵌入缓存
    embedding_generator.save_cache(cache_path)

    logger.info("=" * 50)
    logger.info("ToolBench 数据预处理完成！(G2+G3)")
    logger.info(f"API工具数量: {len(tool_to_idx)}")
    logger.info(f"图边数量: {edge_index.size(1)}")
    logger.info(f"训练样本: {len(train_samples)} 个")
    logger.info(f"验证样本: {len(val_samples)} 个")
    logger.info(f"测试样本: {len(test_samples)} 个")
    logger.info("=" * 50)



class ToolGraphDataset(Dataset):
    """工具图数据集，用于PyTorch DataLoader"""

    def __init__(self, samples_path: Path, full_graph_path: Path):
        """
        初始化数据集

        Args:
            samples_path: 样本文件路径
            full_graph_path: 完整图文件路径
        """
        self.samples = torch.load(samples_path, weights_only=True)
        self.full_graph_data = torch.load(full_graph_path, weights_only=True)

        self.node_features = self.full_graph_data['node_features']
        self.tool_to_idx = self.full_graph_data['tool_to_idx']
        self.full_edge_index = self.full_graph_data['edge_index']
        # 懒缓存：为每个样本缓存子图边，避免每个epoch重复构建，降低CPU占用
        self._edges_cache = [None] * len(self.samples)


        logger.info(f"加载数据集: {len(self.samples)} 个样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本，返回PyG Data对象

        Returns:
            Data: 包含查询节点和工具节点的子图
        """
        sample = self.samples[idx]

        query_embedding = sample['query_embedding']
        subgraph_node_indices = sample['subgraph_node_indices']
        label = sample['label']

        # 提取子图节点特征
        tool_node_features = self.node_features[subgraph_node_indices]

        # 构建子图的边索引
        # 使用样本级缓存，避免重复构建子图边，降低CPU开销
        cached_edges = self._edges_cache[idx]
        if cached_edges is None:
            subgraph_edge_index = self._build_subgraph_edges(subgraph_node_indices)
            self._edges_cache[idx] = subgraph_edge_index
        else:
            subgraph_edge_index = cached_edges

        # 构建完整的节点特征矩阵（查询节点 + 工具节点）
        # 查询节点是节点0，工具节点是节点1到N
        num_tool_nodes = len(subgraph_node_indices)
        x = torch.zeros(num_tool_nodes + 1, query_embedding.size(0))
        x[0] = query_embedding  # 查询节点
        x[1:] = tool_node_features  # 工具节点

        # 构建边索引：查询节点连接到所有工具节点 + 工具节点间的连接
        # 查询节点与工具节点的无向边（双向）
        query_to_tools = torch.stack([
            torch.zeros(num_tool_nodes, dtype=torch.long),  # 查询节点 → 工具节点
            torch.arange(1, num_tool_nodes + 1, dtype=torch.long)
        ])
        tools_to_query = torch.stack([
            torch.arange(1, num_tool_nodes + 1, dtype=torch.long),  # 工具节点 → 查询节点
            torch.zeros(num_tool_nodes, dtype=torch.long)
        ])
        query_edges = torch.cat([query_to_tools, tools_to_query], dim=1)

        # 工具节点间的边（需要重新映射索引）
        if subgraph_edge_index.size(1) > 0:
            # 将原始图中的节点索引映射到子图中的索引（+1因为查询节点占用索引0）
            remapped_edges = subgraph_edge_index + 1
            # 合并查询边和工具边（不需要flip，因为构建图时已经是双向边）
            edge_index = torch.cat([query_edges, remapped_edges], dim=1)
        else:
            edge_index = query_edges

        # 创建PyG Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            y=label,  # 只有工具节点有标签
            num_nodes=num_tool_nodes + 1,
            query_text=sample['query_text']  # 添加query文本用于日志
        )

        return data

    def _build_subgraph_edges(self, subgraph_node_indices):
        """构建子图的边索引（高效张量化版本）"""
        if self.full_edge_index.size(1) == 0 or len(subgraph_node_indices) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        # 创建节点存在标记向量（只为需要的节点范围）
        max_node = max(self.full_edge_index.max().item(), subgraph_node_indices.max().item())
        node_present = torch.zeros(max_node + 1, dtype=torch.bool)
        node_present[subgraph_node_indices] = True

        # 一次性筛选出子图内的边
        edge_mask = node_present[self.full_edge_index[0]] & node_present[self.full_edge_index[1]]

        if not edge_mask.any():
            return torch.empty((2, 0), dtype=torch.long)

        # 提取子图边
        subgraph_full_edges = self.full_edge_index[:, edge_mask]

        # 创建原始索引到子图索引的映射（只为子图节点）
        idx_mapping = torch.full((max_node + 1,), -1, dtype=torch.long)
        idx_mapping[subgraph_node_indices] = torch.arange(len(subgraph_node_indices), dtype=torch.long)

        # 重新映射边索引
        remapped_edges = idx_mapping[subgraph_full_edges]

        return remapped_edges

def create_dataloaders(config, batch_size: int = None, include_test: bool = False):
    """
    创建训练、验证和测试数据加载器

    Args:
        config: 配置字典
        batch_size: 批大小
        include_test: 是否包含测试数据加载器

    Returns:
        train_loader, val_loader: 训练和验证数据加载器
        或 train_loader, val_loader, test_loader: 如果include_test=True
    """
    batch_size = batch_size or config['BATCH_SIZE']

    # 创建数据集
    train_dataset = ToolGraphDataset(config['TRAINING_SAMPLES_PATH'], config['FULL_GRAPH_PATH'])
    val_dataset = ToolGraphDataset(config['VALIDATION_SAMPLES_PATH'], config['FULL_GRAPH_PATH'])
    test_dataset = ToolGraphDataset(config['TEST_SAMPLES_PATH'], config['FULL_GRAPH_PATH'])

    # 创建数据加载器
    # 使用PyG的DataLoader来处理图数据的批处理
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except ImportError:
        # 兼容旧版本PyTorch Geometric
        from torch_geometric.data import DataLoader as PyGDataLoader

    common_kwargs = dict(num_workers=4, persistent_workers=True, pin_memory=True)

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **common_kwargs
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs
    )

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs
    )

    logger.info(f"创建数据加载器: 训练批大小={batch_size}, 验证批大小={batch_size}, 测试批大小={batch_size}")
    return train_loader, val_loader, test_loader

