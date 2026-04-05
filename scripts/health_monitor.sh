#!/bin/bash
# =============================================================================
# RAG Knowledge Base Health Monitor
# =============================================================================
# 定期检查 RAG API / Qdrant / Ollama 三个服务的健康状态
# 如果服务不可达，尝试重启并记录日志
# 
# 使用方式：
#   手动执行：./scripts/health_monitor.sh
#   Cron 定时：每 5 分钟执行一次
#     */5 * * * * /Users/jk/Projects/rag-knowledge-base/scripts/health_monitor.sh >> /Users/jk/Projects/rag-knowledge-base/logs/health_monitor.log 2>&1
# 
# 日志位置：logs/health_monitor.log
# =============================================================================

PROJECT_DIR="/Users/jk/Projects/rag-knowledge-base"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/health_monitor.log"

# 服务配置
API_URL="http://localhost:8000"
QDRANT_URL="http://localhost:6333"
OLLAMA_URL="http://localhost:11434"

# LaunchAgent 服务名
API_SERVICE="com.rag-knowledge-base.api"
QDRANT_SERVICE="com.rag-knowledge-base.qdrant"

# 超时时间（秒）
CHECK_TIMEOUT=5

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 日志函数
log() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# 检查服务健康状态
check_service() {
    local name="$1"
    local url="$2"
    local timeout="$CHECK_TIMEOUT"
    
    # 使用 curl 检查，超时 $timeout 秒
    if curl -sf --max-time "$timeout" "$url" > /dev/null 2>&1; then
        return 0  # 健康
    else
        return 1  # 不健康
    fi
}

# 重启服务
restart_service() {
    local name="$1"
    local launchctl_name="$2"
    
    log "WARN" "服务 $name 不健康，尝试重启..."
    
    # 先停止
    launchctl stop "$launchctl_name" 2>/dev/null
    sleep 2
    
    # 再启动
    if launchctl start "$launchctl_name" 2>/dev/null; then
        log "INFO" "服务 $name 已重启"
        
        # 等待服务启动
        sleep 5
        
        # 验证重启是否成功
        local url=""
        case "$name" in
            "RAG API") url="$API_URL/health" ;;
            "Qdrant") url="$QDRANT_URL/healthz" ;;
        esac
        
        if [ -n "$url" ] && check_service "$name" "$url"; then
            log "INFO" "服务 $name 重启成功，已恢复正常"
            return 0
        else
            log "ERROR" "服务 $name 重启后仍不健康，需要人工介入"
            return 1
        fi
    else
        log "ERROR" "服务 $name 重启失败（launchctl 命令执行失败）"
        return 1
    fi
}

# 检查 Ollama（非 launchctl 管理，只报告）
check_ollama() {
    if check_service "Ollama" "$OLLAMA_URL/api/tags"; then
        log "INFO" "Ollama 服务正常"
        return 0
    else
        log "WARN" "Ollama 服务不健康（非自动管理，需手动重启）"
        log "INFO" "重启命令: ollama serve 或 brew services restart ollama"
        return 1
    fi
}

# 主检查流程
main() {
    log "INFO" "========== 健康检查开始 =========="
    
    local issues=0
    
    # 1. 检查 RAG API
    if check_service "RAG API" "$API_URL/health"; then
        log "INFO" "RAG API 服务正常"
    else
        restart_service "RAG API" "$API_SERVICE"
        issues=$((issues + 1))
    fi
    
    # 2. 检查 Qdrant
    if check_service "Qdrant" "$QDRANT_URL/healthz"; then
        log "INFO" "Qdrant 服务正常"
    else
        restart_service "Qdrant" "$QDRANT_SERVICE"
        issues=$((issues + 1))
    fi
    
    # 3. 检查 Ollama
    if ! check_ollama; then
        issues=$((issues + 1))
    fi
    
    # 总结
    if [ "$issues" -eq 0 ]; then
        log "INFO" "所有服务正常 ✓"
    else
        log "WARN" "发现 $issues 个服务异常，已尝试处理"
    fi
    
    log "INFO" "========== 健康检查结束 =========="
    
    return $issues
}

# 运行主函数
main
exit $?