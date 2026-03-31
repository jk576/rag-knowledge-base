#!/bin/bash
# Embedding Queue Worker 启动脚本

set -e

# 项目路径
PROJECT_DIR="/Users/jk/Projects/rag-knowledge-base"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"
WORKER_SCRIPT="$PROJECT_DIR/scripts/embedding_queue_worker.py"
LOG_DIR="$PROJECT_DIR/logs"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 日志文件
LOG_FILE="$LOG_DIR/embedding-worker.log"

# 记录启动
echo "$(date '+%Y-%m-%d %H:%M:%S') [启动] Embedding Queue Worker" >> "$LOG_FILE"

# 进入项目目录
cd "$PROJECT_DIR"

# 激活虚拟环境并运行 Worker
source "$VENV_ACTIVATE"

# 以守护进程模式运行
python "$WORKER_SCRIPT" --daemon >> "$LOG_FILE" 2>&1

# 记录结束
echo "$(date '+%Y-%m-%d %H:%M:%S') [结束] Worker 停止" >> "$LOG_FILE"