"""
GraphTool主程序 - 简化版
默认运行完整的训练+验证+测试流程
"""
import os
# 关闭 HuggingFace tokenizers 并行相关警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import torch

from data_loader import prepare_data, create_dataloaders
from train import Trainer
from utils import setup_logging, create_model

def create_config(args):
    """根据命令行参数创建配置对象"""
    # 项目根目录
    PROJECT_ROOT = Path(__file__).parent.parent

    # 数据相关路径
    DATASET_DIR = PROJECT_ROOT / "datasets"
    TOOLBENCH_DIR = DATASET_DIR / "ToolBench"
    PREPROCESSED_DATA_DIR = TOOLBENCH_DIR / "preprocessed_data"

    # 输出路径
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    MODELS_DIR = OUTPUTS_DIR / "models"
    LOGS_DIR = OUTPUTS_DIR / "logs"

    # 创建必要的目录
    directories = [DATASET_DIR, TOOLBENCH_DIR, PREPROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR, LOGS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    # 配置字典
    config = {
        # 路径配置
        'PROJECT_ROOT': PROJECT_ROOT,
        'DATASET_DIR': DATASET_DIR,
        'TOOLBENCH_DIR': TOOLBENCH_DIR,
        'PREPROCESSED_DATA_DIR': PREPROCESSED_DATA_DIR,
        'OUTPUTS_DIR': OUTPUTS_DIR,
        'MODELS_DIR': MODELS_DIR,
        'LOGS_DIR': LOGS_DIR,

        'dataset_path': Path(args.dataset_path) if getattr(args, 'dataset_path', None) else (TOOLBENCH_DIR / 'G3_query.json'),
        # 预处理数据文件路径
        'FULL_GRAPH_PATH': PREPROCESSED_DATA_DIR / "full_graph.pt",
        'TRAINING_SAMPLES_PATH': PREPROCESSED_DATA_DIR / "training_samples.pt",
        'VALIDATION_SAMPLES_PATH': PREPROCESSED_DATA_DIR / "validation_samples.pt",
        'TEST_SAMPLES_PATH': PREPROCESSED_DATA_DIR / "test_samples.pt",

        # 模型超参数
        'EMBEDDING_MODEL_NAME': args.embedding_model,
        'EMBEDDING_DIM': args.embedding_dim,
        'HIDDEN_DIM': args.hidden_dim,
        'GNN_TYPE': args.gnn_type,
        'GNN_LAYERS': args.gnn_layers,
        'MLP_HIDDEN_DIM': args.mlp_hidden_dim,
        'DROPOUT_RATE': args.dropout_rate,
        'NUM_HEADS': args.num_heads,

        # 训练超参数
        'LEARNING_RATE': args.lr,
        'WEIGHT_DECAY': args.weight_decay,
        'OPTIMIZER': args.optimizer,
        'BATCH_SIZE': args.batch_size,
        'NUM_EPOCHS': args.epochs,
        'EARLY_STOPPING_PATIENCE': args.early_stopping_patience,
        'TRAIN_SPLIT': args.train_split,
        'VAL_SPLIT': args.val_split,
        'TEST_SPLIT': args.test_split,
        'SAVE_BEST_MODEL': True,  # 默认保存最佳模型

        # 数据预处理参数
        'CO_OCCURRENCE_THRESHOLD': args.co_occurrence_threshold,
        'MAX_SEQUENCE_LENGTH': args.max_sequence_length,
        'EMBED_BATCH_SIZE': args.embed_batch_size,
        'EMBED_DTYPE': args.embed_dtype,
        'RANDOM_SEED': 42,  # 默认不固定种子

        # 设备配置 (将在后面处理)
        'DEVICE': args.device,
        'DEFAULT_DEVICE': args.device,

        # 日志配置
        'LOG_LEVEL': 'INFO',  # 默认INFO级别
        'LOG_FORMAT': "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }

    # 处理设备配置
    device_str = args.device.lower()

    # 检查是否为数字（GPU ID）
    if device_str.isdigit():
        gpu_id = int(device_str)
        if not torch.cuda.is_available():
            raise ValueError("CUDA不可用，无法使用GPU设备")
        gpu_count = torch.cuda.device_count()
        if gpu_id >= gpu_count:
            raise ValueError(f"指定的GPU ID {gpu_id} 超出可用GPU数量 {gpu_count}")
        config['DEVICE'] = f"cuda:{gpu_id}"
        config['DEFAULT_DEVICE'] = f"cuda:{gpu_id}"
        config['GPU_COUNT'] = gpu_count
        config['GPU_ID'] = gpu_id

    elif device_str == "cuda":
        if torch.cuda.is_available():
            config['GPU_COUNT'] = torch.cuda.device_count()
            config['GPU_ID'] = None  # 使用默认GPU
        else:
            raise ValueError("CUDA不可用，请使用CPU或检查CUDA安装")
    elif device_str == "cpu":
        config['DEVICE'] = 'cpu'
        config['DEFAULT_DEVICE'] = 'cpu'
        config['GPU_COUNT'] = 0
        config['GPU_ID'] = None
    else:
        raise ValueError(f"不支持的设备类型: {device_str}，支持的类型: cpu, cuda, 0, 1, 2...")

    return config

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GraphTool: 查询感知的图神经网络API选择模型")

    # 数据参数
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批大小 (默认: 32)')
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='G3_query.json 路径，优先于默认 datasets/ToolBench/G3_query.json')

    # 模型参数
    parser.add_argument('--gnn-type', type=str, default='GCN',
                       choices=['GCN', 'GAT'],
                       help='GNN类型 (默认: GCN)')
    parser.add_argument('--embedding-model', type=str, default="Qwen/Qwen3-Embedding-8B",
                       help='预训练嵌入模型 (默认: Qwen/Qwen3-Embedding-8B)')
    parser.add_argument('--embedding-dim', type=int, default=4096,
                       help='嵌入维度 (默认: 4096)')
    parser.add_argument('--hidden-dim', type=int, default=4096,
                       help='GNN隐藏层维度 (默认: 4096)')
    parser.add_argument('--gnn-layers', type=int, default=2,
                       help='GNN层数 (默认: 2)')
    parser.add_argument('--mlp-hidden-dim', type=int, default=512,
                       help='输出MLP隐藏层维度 (默认: 512)')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                       help='Dropout率 (默认: 0.0)')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='GAT注意力头数 (默认: 4)')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=3,
                       help='训练轮数 (默认: 3)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率 (默认: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0,
                       help='L2正则化参数 (默认: 0)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                       help='优化器类型 (默认: Adam)')
    parser.add_argument('--early-stopping-patience', type=int, default=2,
                       help='早停耐心值 (默认: 2)')
    parser.add_argument('--train-split', type=float, default=0.7,
                       help='训练集比例 (默认: 0.7)')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='测试集比例 (默认: 0.2)')

    # 数据预处理参数
    parser.add_argument('--co-occurrence-threshold', type=int, default=2,
                       help='共现阈值K，用于创建边 (默认: 2)')
    parser.add_argument('--max-sequence-length', type=int, default=512,
                       help='文本最大长度 (默认: 512)')
    parser.add_argument('--embed-batch-size', type=int, default=1,
                       help='预处理时生成嵌入的批大小 (默认: 1)')
    parser.add_argument('--embed-dtype', type=str, choices=['fp16', 'fp32'], default='fp32',
                       help='预处理嵌入模型权重精度: fp16 或 fp32 (默认: fp16)')

    # 设备参数
    parser.add_argument('--device', type=str, default='0',
                       help='训练设备: cpu, cuda, 0, 1, 2... (数字表示指定GPU ID, 默认: 0)')

    # 其他参数
    parser.add_argument('--threshold-mode', type=str, default='query', choices=['fixed', 'query'],
                       help='阈值模式: fixed(固定0.5) 或 query(动态query阈值) (默认: query)')

    # 输出参数
    parser.add_argument('--model-name', type=str, default='query_aware_gnn',
                       help='模型保存名称 (默认: query_aware_gnn)')
    parser.add_argument('--log-name', type=str, default='debug',
                       help='日志文件名称 (默认: debug)')

    return parser.parse_args()

def main():
    # 解析参数
    args = parse_arguments()

    # 创建配置
    config = create_config(args)

    # 设置日志
    logger = setup_logging(config, args.log_name)

    # 打印配置信息
    gpu_info = f"GPU-{config.get('GPU_ID')}" if config.get('GPU_ID') is not None else "auto"
    logger.info(f"Config: Device={config['DEVICE']} ({gpu_info}), Batch={args.batch_size}, GNN={config['GNN_TYPE']}, "
               f"Layers={config['GNN_LAYERS']}, Hidden={config['HIDDEN_DIM']}, Epochs={args.epochs}, LR={config['LEARNING_RATE']}, "
               f"WD={config['WEIGHT_DECAY']}, Dropout={config['DROPOUT_RATE']}, Threshold={args.threshold_mode}")



    try:
        # 自动检测是否需要预处理数据
        need_preprocessing = not all([
            config['FULL_GRAPH_PATH'].exists(),
            config['TRAINING_SAMPLES_PATH'].exists(),
            config['VALIDATION_SAMPLES_PATH'].exists(),
            config['TEST_SAMPLES_PATH'].exists()
        ])

        if need_preprocessing:
            logger.info("检测到缺少预处理数据，开始数据预处理...")
            prepare_data(config)
        else:
            logger.info("检测到已有预处理数据，跳过数据预处理步骤")

        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            config,
            batch_size=args.batch_size,
            include_test=True
        )

        # 创建模型
        model = create_model(config)

        # 创建训练器
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device=config['DEVICE'],  # 使用处理后的设备字符串
            logger=logger,
            threshold_mode=args.threshold_mode
        )

        # 开始训练
        trainer.train(num_epochs=args.epochs)



    except Exception as e:
        logger.error(f"程序执行出错: {e}")
        raise

if __name__ == "__main__":
    main()
