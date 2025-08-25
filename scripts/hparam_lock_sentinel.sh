#!/usr/bin/env bash
set -euo pipefail

LOCK="outputs/hparam_search_running.lock"
mkdir -p outputs

echo "[sentinel] 启动，维护锁文件: $LOCK"

while true; do
  # 认为以下任一进程存在即表示搜参/训练在进行：
  if pgrep -f "scripts/hyperparameter_search.sh" >/dev/null \
     || pgrep -f "python[3]* .*src/main.py" >/dev/null; then
    # 写入锁（幂等）
    date > "$LOCK"
  else
    # 删除锁（幂等）
    rm -f "$LOCK"
  fi
  sleep 20
done

