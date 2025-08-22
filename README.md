## 项目结构

```
GraphTool/
├── datasets/                   # 原始数据集
│   └── API-BANK/              # API-BANK数据集
├── preprocessed_data/          # 预处理后的数据
│   ├── full_graph.pt          # 完整工具图
│   ├── training_samples.pt    # 训练样本
│   └── validation_samples.pt  # 验证样本
├── outputs/                    # 输出文件
│   ├── models/                # 保存的模型
│   └── logs/                  # 训练日志
├── src/                       # 源代码
│   ├── __init__.py
│   ├── config.py              # 配置文件
│   ├── data_loader.py         # 数据加载和预处理
│   ├── model.py               # GNN模型定义
│   ├── train.py               # 训练逻辑
│   ├── main.py                # 主程序入口
│   └── utils.py               # 工具函数
├── requirements.txt           # 项目依赖
├── test_setup.py             # 项目设置测试
└── README.md                 # 项目说明
```

## 安装和设置

### 1. 克隆项目
```bash
git clone <repository-url>
cd GraphTool
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
python test_setup.py
```

## 使用方法

### 1. 数据准备
将API-BANK数据集放置在 `datasets/API-BANK/` 目录下。

### 2. 训练模型
```bash
# 基础训练
python src/main.py --mode train

# 自定义参数训练
python src/main.py --mode train \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.001 \
    --gnn-type GAT \
    --model-type attention
```

### 3. 评估模型
```bash
python src/main.py --mode eval --model-name your_model_name
```

