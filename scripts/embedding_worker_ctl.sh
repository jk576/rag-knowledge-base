#!/bin/bash
# Embedding Worker 管理脚本

set -e

PROJECT_DIR="/Users/jk/Projects/rag-knowledge-base"
PLIST_NAME="com.rag-knowledge-base.embedding-worker"
PLIST_PATH="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
LOG_FILE="$PROJECT_DIR/logs/embedding-worker.log"

# 命令
case "$1" in
    start)
        echo "启动 Embedding Worker..."
        launchctl load "$PLIST_PATH"
        echo "已加载 LaunchAgent"
        ;;
    
    stop)
        echo "停止 Embedding Worker..."
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        echo "已停止"
        ;;
    
    restart)
        echo "重启 Embedding Worker..."
        launchctl unload "$PLIST_PATH" 2>/dev/null || true
        sleep 2
        launchctl load "$PLIST_PATH"
        echo "已重启"
        ;;
    
    status)
        echo "检查 Embedding Worker 状态..."
        launchctl list | grep "$PLIST_NAME" || echo "未运行"
        
        echo ""
        echo "最近日志（最后 20 行）："
        if [ -f "$LOG_FILE" ]; then
            tail -20 "$LOG_FILE"
        else
            echo "日志文件不存在"
        fi
        ;;
    
    logs)
        echo "查看完整日志..."
        if [ -f "$LOG_FILE" ]; then
            less "$LOG_FILE"
        else
            echo "日志文件不存在"
        fi
        ;;
    
    queue)
        echo "查看队列状态..."
        cd "$PROJECT_DIR"
        source .venv/bin/activate
        python -c "
from src.core.embedding_queue import get_queue_manager
import json
status = get_queue_manager().get_queue_status()
print(json.dumps(status, indent=2))
"
        ;;
    
    manual)
        echo "手动运行 Worker（一次性）..."
        cd "$PROJECT_DIR"
        source .venv/bin/activate
        python scripts/embedding_queue_worker.py
        ;;
    
    *)
        echo "用法: $0 {start|stop|restart|status|logs|queue|manual}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动 Worker LaunchAgent"
        echo "  stop    - 停止 Worker"
        echo "  restart - 重启 Worker"
        echo "  status  - 查看状态和最近日志"
        echo "  logs    - 查看完整日志"
        echo "  queue   - 查看队列状态"
        echo "  manual  - 手动运行一次（不使用 LaunchAgent）"
        exit 1
        ;;
esac