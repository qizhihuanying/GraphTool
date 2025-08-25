#!/bin/bash

# GraphTool 超参数搜索脚本
# 支持GPU队列管理和动态任务调度

# 超参数配置
LR_VALUES=(1e-3 3e-3 1e-4 3e-4 1e-5 3e-5 1e-6 3e-6)
L2_VALUES=(0 1e-7 1e-6 1e-5)
GNN_TYPES=(GCN)
LAYER_VALUES=(6 7 8 9 10)
EPOCHS=5

# GPU配置 - 用户可修改这个数组来控制并发数
# 例如: GPUS=(6 6 6 7 7 7) 表示GPU6和GPU7各跑3个并发任务
GPUS=(0 0 0 1 1 1 2 2 2 4 4 4 5 5 5 6 6 6 7 7 7)  # 默认配置，用户可根据需要修改

# 全局变量
declare -a RUNNING_JOBS=()  # 存储正在运行的任务PID
declare -a GPU_USAGE=()    # 记录每个GPU槽位的使用状态
declare -a ALL_EXPERIMENTS=()  # 所有实验配置

# 初始化GPU使用状态
init_gpu_usage() {
    for i in "${!GPUS[@]}"; do
        GPU_USAGE[$i]=0  # 0表示空闲，1表示占用
    done
}

# 生成所有实验配置
generate_experiments() {
    echo "正在生成实验配置..."
    local count=0

    for lr in "${LR_VALUES[@]}"; do
        for l2 in "${L2_VALUES[@]}"; do
            for gnn_type in "${GNN_TYPES[@]}"; do
                for layers in "${LAYER_VALUES[@]}"; do
                    local exp_name="lr=${lr}_l2=${l2}_type=${gnn_type}_layer=${layers}"
                    ALL_EXPERIMENTS+=("$lr|$l2|$gnn_type|$layers|$exp_name")
                    ((count++))
                done
            done
        done
    done

    echo "总共生成 $count 个实验配置"
    echo "GPU配置: ${GPUS[*]} (总共 ${#GPUS[@]} 个并发槽位)"
    echo ""
}

# 检查是否有空闲的GPU槽位
get_free_gpu_slot() {
    for i in "${!GPU_USAGE[@]}"; do
        if [[ ${GPU_USAGE[$i]} -eq 0 ]]; then
            echo $i
            return 0
        fi
    done
    echo -1  # 没有空闲槽位
}

# 启动单个实验
start_experiment() {
    local gpu_slot=$1
    local exp_config=$2

    # 解析实验配置
    IFS='|' read -r lr l2 gnn_type layers exp_name <<< "$exp_config"
    local gpu_id=${GPUS[$gpu_slot]}

    # 创建日志文件路径
    local log_file="outputs/logs/${exp_name}.log"

    # 构建命令
    local cmd="python src/main.py --device $gpu_id --lr $lr --weight-decay $l2 --gnn-type $gnn_type --gnn-layers $layers --epochs $EPOCHS --log-name ${exp_name} --model-name ${exp_name}"

    echo "[$(date '+%H:%M:%S')] 启动实验: $exp_name (GPU $gpu_id, 槽位 $gpu_slot)"
    echo "命令: $cmd"
    echo "日志: $log_file"
    echo "注意: 训练进度条将显示在此终端，关键日志写入上述文件"
    echo "----------------------------------------"

    # 启动实验（后台运行）
    # 设置环境变量，让日志写入指定文件，进度条显示在终端
    (
        export EXTRA_LOG_FILE="$log_file"
        echo ""
        echo "🚀 [实验开始] $exp_name - GPU $gpu_id"
        echo "========================================"
        $cmd

        # 实验完成后立即清理checkpoint文件以节省磁盘空间
        checkpoint_file="outputs/models/${exp_name}_best.pt"
        if [[ -f "$checkpoint_file" ]]; then
            file_size=$(du -h "$checkpoint_file" | cut -f1)
            rm -f "$checkpoint_file"
            echo "🗑️  已清理checkpoint: ${exp_name}_best.pt (节省 $file_size)"
        fi

        echo "========================================"
        echo "✅ [实验完成] $exp_name - GPU $gpu_id"
        echo ""
    ) &
    local job_pid=$!

    # 记录任务信息
    RUNNING_JOBS[$gpu_slot]=$job_pid
    GPU_USAGE[$gpu_slot]=1

    return 0
}

# 检查并清理完成的任务
check_completed_jobs() {
    local completed_count=0

    for i in "${!RUNNING_JOBS[@]}"; do
        local pid=${RUNNING_JOBS[$i]}
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            # 任务已完成
            local gpu_id=${GPUS[$i]}
            echo ""
            echo "[$(date '+%H:%M:%S')] GPU $gpu_id 槽位 $i 的任务已完成 (PID: $pid)"

            # 清理状态
            RUNNING_JOBS[$i]=""
            GPU_USAGE[$i]=0
            ((completed_count++))
        fi
    done

    if [[ $completed_count -gt 0 ]]; then
        echo "释放了 $completed_count 个GPU槽位"
        echo ""
    fi
}

# 显示当前状态
show_status() {
    local running_count=0
    local completed_count=0

    for status in "${GPU_USAGE[@]}"; do
        if [[ $status -eq 1 ]]; then
            ((running_count++))
        fi
    done

    completed_count=$((${#ALL_EXPERIMENTS[@]} - ${#REMAINING_EXPERIMENTS[@]} - running_count))

    echo "========== 当前状态 =========="
    echo "总实验数: ${#ALL_EXPERIMENTS[@]}"
    echo "已完成: $completed_count"
    echo "正在运行: $running_count"
    echo "等待中: ${#REMAINING_EXPERIMENTS[@]}"
    echo "GPU使用情况:"
    for i in "${!GPUS[@]}"; do
        local gpu_id=${GPUS[$i]}
        local status_text="空闲"
        if [[ ${GPU_USAGE[$i]} -eq 1 ]]; then
            status_text="占用 (PID: ${RUNNING_JOBS[$i]})"
        fi
        echo "  GPU $gpu_id 槽位 $i: $status_text"
    done
    echo "============================="
    echo ""
}

# 主循环
main() {
    echo "GraphTool 超参数搜索启动"
    echo "=========================="

    # 创建日志目录和锁文件目录
    mkdir -p outputs/logs
    mkdir -p outputs

    # 创建搜参锁文件，避免其他进程（如 o.py --auto）在期间占用GPU
    HPARAM_LOCK="outputs/hparam_search_running.lock"
    echo "$(date)" > "$HPARAM_LOCK"

    # 初始化
    init_gpu_usage
    generate_experiments

    # 复制实验列表到待处理队列
    declare -a REMAINING_EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")

    echo "开始执行实验..."
    echo ""

    # 主循环
    while [[ ${#REMAINING_EXPERIMENTS[@]} -gt 0 ]] || [[ $(echo "${GPU_USAGE[@]}" | grep -o "1" | wc -l) -gt 0 ]]; do
        # 检查完成的任务
        check_completed_jobs

        # 启动新任务（如果有空闲槽位和待处理实验）
        while [[ ${#REMAINING_EXPERIMENTS[@]} -gt 0 ]]; do
            local free_slot=$(get_free_gpu_slot)
            if [[ $free_slot -eq -1 ]]; then
                break  # 没有空闲槽位
            fi

            # 取出第一个待处理实验
            local next_exp="${REMAINING_EXPERIMENTS[0]}"
            REMAINING_EXPERIMENTS=("${REMAINING_EXPERIMENTS[@]:1}")  # 移除第一个元素

            # 启动实验
            start_experiment $free_slot "$next_exp"
        done

        # 显示状态
        show_status

        # 等待一段时间再检查
        sleep 30
    done

    echo "所有实验已完成！"

    # 删除搜参锁文件，允许 o.py --auto 开始检测并占用GPU
    if [[ -f "$HPARAM_LOCK" ]]; then
        rm -f "$HPARAM_LOCK"
        echo "已删除锁文件: $HPARAM_LOCK"
    fi

    # 汇总 TOP-3（按测试集 R@5）
    echo "\n基于测试 R@5 的 TOP-3:"
    declare -a SCORED=()
    for exp_config in "${ALL_EXPERIMENTS[@]}"; do
        IFS='|' read -r lr l2 gnn_type layers exp_name <<< "$exp_config"
        local log_file="outputs/logs/${exp_name}.log"
        if [[ -f "$log_file" ]]; then
            # 解析最后一行包含 Test Loss 和指标的行，提取 R@5
            local last_test_line=$(grep -E "Test Loss:" "$log_file" | tail -n 1)
            if [[ -n "$last_test_line" ]]; then
                # 示例：Test Loss: 0.1234, R@3: 0.11, R@5: 0.22, N@3: ..., N@5: ..., C@3: ..., C@5: ...
                local r5=$(echo "$last_test_line" | sed -n 's/.*R@5: \([0-9.][0-9.]*\).*/\1/p')
                if [[ -n "$r5" ]]; then
                    SCORED+=("$r5|$exp_name")
                fi
            fi
        fi
    done
    if [[ ${#SCORED[@]} -eq 0 ]]; then
        echo "未找到可解析的测试 R@5 结果"
    else
        printf '%s\n' "${SCORED[@]}" | sort -t '|' -k1,1nr | head -n 3 | awk -F '|' '{printf("  #%d %s (R@5=%.4f)\n", NR, $2, $1)}'
    fi

    echo "日志文件位于: outputs/logs/"
    echo ""

    # 显示结果摘要
    echo "实验结果摘要:"
    echo "============"
    for exp_config in "${ALL_EXPERIMENTS[@]}"; do
        IFS='|' read -r lr l2 gnn_type layers exp_name <<< "$exp_config"
        local log_file="outputs/logs/${exp_name}.log"
        if [[ -f "$log_file" ]]; then
            echo "✓ $exp_name"
        else
            echo "✗ $exp_name (日志文件不存在)"
        fi
    done
}

# 信号处理：优雅退出
cleanup() {
    echo ""
    echo "收到退出信号，正在清理..."

    # 终止所有子进程
    for pid in "${RUNNING_JOBS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "终止任务 PID: $pid"
            kill "$pid"
        fi
    done

    # 清理锁文件
    if [[ -f "$HPARAM_LOCK" ]]; then
        rm -f "$HPARAM_LOCK"
        echo "已删除锁文件: $HPARAM_LOCK"
    fi

    echo "清理完成"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 检查参数
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    echo "GraphTool 超参数搜索脚本"
    echo ""
    echo "使用方法:"
    echo "  $0                    # 使用默认GPU配置运行"
    echo "  $0 --help            # 显示帮助信息"
    echo ""
    echo "GPU配置:"
    echo "  修改脚本中的 GPUS 数组来控制并发数"
    echo "  例如: GPUS=(6 6 6 7 7 7) 表示GPU6和GPU7各跑3个并发任务"
    echo ""
    echo "超参数配置:"
    echo "  LR: ${LR_VALUES[*]}"
    echo "  L2: ${L2_VALUES[*]}"
    echo "  GNN类型: ${GNN_TYPES[*]}"
    echo "  层数: ${LAYER_VALUES[*]}"
    echo "  训练轮数: $EPOCHS"
    echo ""
    exit 0
fi

# 运行主程序
main
