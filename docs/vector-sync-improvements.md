# 向量化重试与同步机制改进方案

## 一、问题分析

### 1.1 向量化失败重试问题

**现有机制**：
- `embedding.py`: tenacity retry，最多 3 次，只 retry 网络错误
- `embedding_queue_worker.py`: 队列模式有 `retry_count` 字段
- 直接模式（非队列）无重试机制

**缺陷**：
- 队列模式下失败项重试 3 次后就放弃
- 直接模式（`_vectorize_and_save_chunks`）无重试
- 失败状态未持久化到 chunks 表
- 无自动恢复机制

### 1.2 Qdrant 向量同步问题

**已发现问题**：
- 易学资料：28 个完成文档向量缺失（写入失败）
- openviking：121 个孤儿向量（删除未清理）

**根本原因**：
- Qdrant upsert 可能部分失败
- 删除文档时向量删除失败未清理
- 无定期一致性检查机制

---

## 二、改进方案

### 2.1 向量化失败重试改进

#### 2.1.1 chunks 表扩展

```sql
ALTER TABLE chunks ADD COLUMN vector_status TEXT DEFAULT 'pending';
-- 状态: pending, success, failed
ALTER TABLE chunks ADD COLUMN vector_error TEXT;
ALTER TABLE chunks ADD COLUMN retry_count INTEGER DEFAULT 0;
ALTER TABLE chunks ADD COLUMN last_vector_attempt TIMESTAMP;
```

#### 2.1.2 向量化服务改进

```python
# src/core/embedding.py 增强

class EmbeddingService:
    # 增加持久化重试状态
    def embed_with_retry_tracking(
        self, 
        text: str, 
        chunk_id: str,
        db_path: Path
    ) -> tuple[Optional[List[float]], str]:
        """
        带状态跟踪的向量化
        
        Returns:
            (embedding, status) - status: 'success' | 'failed' | 'will_retry'
        """
        pass
```

#### 2.1.3 失败项自动重试

新增脚本: `scripts/retry_failed_vectors.py`

```python
"""
重试向量化失败的 chunks

流程：
1. 查询 vector_status='failed' AND retry_count < MAX_RETRY 的 chunks
2. 重置为 pending
3. 重新加入队列
"""
```

#### 2.1.4 API 接口

```python
# 新增查询接口
GET /api/vectors/failed?project_id=xxx&limit=50
# 返回失败的 chunks 列表

POST /api/vectors/retry
# 重试失败的向量
{
    "project_id": "xxx",  # 可选，不填则全部项目
    "chunk_ids": ["id1", "id2"],  # 可选
    "reset_all": false  # 是否重置所有失败项
}
```

---

### 2.2 Qdrant 向量同步改进

#### 2.2.1 删除文档时确保向量清理

```python
# src/services/document_service.py 改进

def delete_document(self, document_id: str, delete_file: bool = True) -> bool:
    # 1. 收集所有向量 ID
    # 2. 批量删除 Qdrant 向量（带重试）
    # 3. 记录删除失败的向量 ID
    # 4. 写入 orphan_vectors 表待后续清理
    # 5. 删除 SQLite 记录
    pass
```

#### 2.2.2 孤儿向量表

```sql
CREATE TABLE orphan_vectors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    vector_id TEXT NOT NULL,
    chunk_id TEXT,  -- 可能为空（chunk 已删除）
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    cleaned_at TIMESTAMP,
    
    UNIQUE(project_id, vector_id)
);
```

#### 2.2.3 一致性检查脚本增强

增强 `scripts/sync_missing_vectors.py`:

```python
def check_consistency(project_id: str) -> dict:
    """
    检查 SQLite 和 Qdrant 一致性
    
    Returns:
        {
            "sqlite_count": int,
            "qdrant_count": int,
            "missing_in_qdrant": [vector_id, ...],  # SQLite 有但 Qdrant 无
            "orphan_in_qdrant": [vector_id, ...],   # Qdrant 有但 SQLite 无
            "action_needed": str
        }
    """
    pass

def fix_orphan_vectors(project_id: str, dry_run: bool = True) -> int:
    """
    清理孤儿向量
    
    Args:
        dry_run: 只报告不执行
        
    Returns:
        清理的向量数
    """
    pass
```

#### 2.2.4 定期同步任务

```python
# 新增: scripts/cron_vector_sync.py

"""
Cron 定期同步任务

建议频率: 每日凌晨 3 点

任务：
1. 检查所有项目的向量一致性
2. 同步缺失的向量
3. 清理孤儿向量
4. 重试失败的向量
5. 生成报告
"""
```

---

## 三、实施计划

### Phase 1: 数据库扩展（预计 30 分钟）

- [ ] 添加 chunks 表字段
- [ ] 创建 orphan_vectors 表
- [ ] 数据迁移：更新现有 chunks 的 vector_status

### Phase 2: 向量化重试改进（预计 1 小时）

- [ ] 修改 `embedding.py` 添加状态跟踪
- [ ] 修改 `document_service.py` 使用新字段
- [ ] 修改 `embedding_queue_worker.py` 更新新字段
- [ ] 创建 `scripts/retry_failed_vectors.py`

### Phase 3: 向量同步改进（预计 1 小时）

- [ ] 修改 `document_service.delete_document()` 改进删除逻辑
- [ ] 增强 `scripts/sync_missing_vectors.py` 添加孤儿清理
- [ ] 创建 `scripts/cron_vector_sync.py`

### Phase 4: API 接口（预计 30 分钟）

- [ ] 添加失败向量查询 API
- [ ] 添加重试 API

### Phase 5: 测试验证（预计 30 分钟）

- [ ] 单元测试
- [ ] 集成测试
- [ ] 手动验证

---

## 四、验证标准

1. **重试机制**：
   - 向量化失败后 `vector_status='failed'`
   - `retry_failed_vectors.py` 可以重置并重试
   - API 可以查询和触发重试

2. **同步机制**：
   - 删除文档后无孤儿向量残留
   - 一致性检查可以发现差异
   - 同步脚本可以修复差异

3. **监控**：
   - 定期任务生成报告
   - 可以查看各项目的向量状态

---

## 五、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 数据库迁移失败 | 高 | 备份数据库，使用事务 |
| 大量重试导致 Ollama 过载 | 中 | 控制并发，添加延迟 |
| 孤儿向量误删 | 中 | dry_run 先检查，确认后执行 |
| 向量状态不一致 | 低 | 一致性检查脚本兜底 |

---

*Created: 2026-04-01*