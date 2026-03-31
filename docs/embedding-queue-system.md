# Embedding Queue System - 完整实现

## 概述

队列化 Embedding 系统，解决 Ollama 在本机多模型环境下的并发不稳定问题。

## 问题背景

本机硬件有限 + 多模型共存：
- `qwen3:8b` (解析模型/VLM)
- `bge-m3` (向量模型)
- `qwen2.5:7b` (备选)

Ollama 并发请求会导致：
- 服务不稳定
- 响应超时
- 内存压力
- 古籍内容 token 消耗高（1汉字≈2-3 tokens）

## 解决方案

队列化处理：新文档入库时将 chunks 写入队列，Worker 后台逐个处理。

```
┌─────────────────────────────────────────────────────────────┐
│  新文档入库流程                                              │
│                                                              │
│  文件解析 → 分块 → 保存 chunks → 写入 embedding_queue         │
│                              ↓                               │
│                     ┌────────────────┐                       │
│                     │ Queue Worker   │                       │
│                     │ (独立进程)      │                       │
│                     ├────────────────┤                       │
│                     │ 1. 取 pending   │                       │
│                     │ 2. 调用 Ollama  │                       │
│                     │ 3. 延迟 200ms   │                       │
│                     │ 4. 写入 Qdrant  │                       │
│                     │ 5. 更新状态     │                       │
│                     │ 6. 循环         │                       │
│                     └────────────────┘                       │
│                              ↓                               │
│                     chunks.vector_id 更新                    │
│                     Qdrant 向量库同步                         │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. EmbeddingQueueManager (`src/core/embedding_queue.py`)

队列管理接口，供 DocumentService 使用：

```python
from src.core.embedding_queue import get_queue_manager

manager = get_queue_manager()

# 加入队列
manager.queue_chunks([{"id": chunk_id, "content": content}])

# 查看状态
status = manager.get_queue_status()
# {"pending": 10, "done": 100, "failed": 2, "coverage": 98.5}
```

### 2. EmbeddingQueueWorker (`scripts/embedding_queue_worker.py`)

Worker 进程，后台处理队列：

```bash
# 一次性运行
python scripts/embedding_queue_worker.py

# 守护进程模式
python scripts/embedding_queue_worker.py --daemon

# 限制处理数
python scripts/embedding_queue_worker.py --max 100
```

### 3. DocumentService 集成

新文档入库时自动使用队列：

```python
# 队列模式（默认）
service = DocumentService(db, use_queue=True)

# 直接模式（重新索引等）
service = DocumentService(db, use_queue=False)
```

### 4. LaunchAgent 后台服务

Worker 作为系统服务自动启动：

```bash
# 启动
launchctl load ~/Library/LaunchAgents/com.rag-knowledge-base.embedding-worker.plist

# 停止
launchctl unload ~/Library/LaunchAgents/com.rag-knowledge-base.embedding-worker.plist

# 查看状态
launchctl list | grep embedding-worker
```

## 管理脚本

`scripts/embedding_worker_ctl.sh` 提供便捷控制：

```bash
# 启动 LaunchAgent
./scripts/embedding_worker_ctl.sh start

# 停止
./scripts/embedding_worker_ctl.sh stop

# 重启
./scripts/embedding_worker_ctl.sh restart

# 查看状态
./scripts/embedding_worker_ctl.sh status

# 查看队列
./scripts/embedding_worker_ctl.sh queue

# 手动运行
./scripts/embedding_worker_ctl.sh manual
```

## 配置参数

| 参数 | 值 | 说明 |
|------|---|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `OLLAMA_MODEL` | `bge-m3` | 向量化模型 |
| `MAX_CHARS` | `4000` | 内容截断上限 |
| `PROCESS_DELAY` | `0.2s` | 每请求延迟 |
| `MAX_RETRY` | `3` | 最大重试次数 |

## 数据表结构

### embedding_queue

```sql
CREATE TABLE embedding_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL UNIQUE,
    content TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending/processing/done/failed
    retry_count INTEGER DEFAULT 0,
    error_msg TEXT,
    vector_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);
```

## 当前状态

| 指标 | 值 |
|------|---|
| 总 chunks | 21,688 |
| 向量覆盖率 | **100%** |
| 队列 pending | 0 |
| 队列 failed | 0 |

## 文件清单

```
/Users/jk/Projects/rag-knowledge-base/
├── src/core/
│   ├── embedding_queue.py         # 队列管理器
│   └── embedding.py               # Embedding 服务（含 retry）
├── src/services/
│   └── document_service.py        # 文档服务（集成队列）
├── scripts/
│   ├── embedding_queue_worker.py  # Worker 进程
│   ├── embedding_worker_ctl.sh    # 管理脚本
│   └── start_embedding_worker.sh  # 启动脚本
├── logs/
│   ├── embedding-worker.log       # Worker 日志
│   └── embedding-worker-error.log # 错误日志
└── db/
    └── metadata.db                # 数据库（含 embedding_queue 表）

~/Library/LaunchAgents/
└── com.rag-knowledge-base.embedding-worker.plist  # LaunchAgent 配置
```

## 使用流程

### 新文档入库（自动）

1. 文件同步检测到新文件
2. DocumentService 处理文档
3. 分块保存到 chunks 表
4. 自动加入 embedding_queue
5. Worker 后台处理
6. 向量写入 Qdrant
7. chunks.vector_id 更新

### 手动触发

```bash
# 查看队列状态
./scripts/embedding_worker_ctl.sh queue

# 手动处理
./scripts/embedding_worker_ctl.sh manual

# 启动后台服务
./scripts/embedding_worker_ctl.sh start
```

## 未来优化

- [ ] 添加 Prometheus metrics 监控
- [ ] 支持 webhook 通知（处理完成）
- [ ] 批量向量化优化（多 chunks 合并请求）
- [ ] 智能延迟调整（根据 Ollama 响应时间）
- [ ] 研究 Late Chunking 实现（Jina AI）

---
Created: 2026-03-31