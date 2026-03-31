"""Ollama Embedding Queue Worker

队列化向量化服务，确保 Ollama 不会被并发请求冲击。

完整流程：
1. 所有向量化请求写入 SQLite 队列表
2. 单独 Worker 进程逐个处理
3. 向量化成功后写入 Qdrant 向量库
4. 更新 chunks 表的 vector_id
5. 支持 retry 和失败记录
"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

import sqlite3
import json
import httpx
import time
import signal
import logging
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("embedding-queue")

# 配置
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "bge-m3"
MAX_CHARS = 4000
PROCESS_DELAY = 0.2  # 每个请求后延迟 200ms（更保守）
MAX_RETRY = 3
DB_PATH = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")


@dataclass
class QueueItem:
    """队列项"""
    id: int
    chunk_id: str
    content: str
    status: str  # pending, processing, done, failed
    retry_count: int
    error_msg: Optional[str]
    created_at: str
    processed_at: Optional[str]


class EmbeddingQueue:
    """Embedding 队列管理"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_queue_table()
        self._running = True
        self.client = httpx.Client(timeout=30.0)
        self.vector_store = None  # 延迟初始化
    
    def _init_queue_table(self):
        """初始化队列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='embedding_queue'
        """)
        
        if not cursor.fetchone():
            # 创建队列表
            cursor.execute("""
                CREATE TABLE embedding_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    retry_count INTEGER DEFAULT 0,
                    error_msg TEXT,
                    vector_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed_at TIMESTAMP,
                    
                    UNIQUE(chunk_id)
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_status 
                ON embedding_queue(status, retry_count)
            """)
            
            conn.commit()
            logger.info("创建 embedding_queue 表")
        
        conn.close()
    
    def _get_vector_store(self):
        """延迟初始化 VectorStore"""
        if self.vector_store is None:
            from src.core.vector_store import VectorStore
            self.vector_store = VectorStore()
        return self.vector_store
    
    def populate_queue(self):
        """从 chunks 表填充队列（无向量的 chunks）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 查找无向量的 chunks
        cursor.execute("""
            SELECT id, content
            FROM chunks
            WHERE vector_id IS NULL
            AND id NOT IN (SELECT chunk_id FROM embedding_queue WHERE status != 'failed')
        """)
        
        chunks = cursor.fetchall()
        
        if not chunks:
            logger.info("队列已满，无新 chunks")
            conn.close()
            return 0
        
        # 插入队列
        inserted = 0
        for chunk_id, content in chunks:
            # 截断保护
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]
            
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO embedding_queue (chunk_id, content)
                    VALUES (?, ?)
                """, (chunk_id, content))
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.IntegrityError:
                pass
        
        conn.commit()
        conn.close()
        
        logger.info(f"添加 {inserted} 个 chunks 到队列")
        return inserted
    
    def get_chunk_info(self, chunk_id: str) -> Optional[dict]:
        """获取 chunk 的完整信息（包括 project_id, document_id）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.document_id, c.project_id, c.content, d.filename, d.source_path, c.metadata_json
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.id = ?
        """, (chunk_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "document_id": row[0],
                "project_id": row[1],
                "content": row[2],
                "filename": row[3] or "unknown",
                "source_path": row[4],
                "metadata_json": row[5]
            }
        return None
    
    def get_pending_item(self) -> Optional[QueueItem]:
        """获取下一个待处理项"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, chunk_id, content, status, retry_count, error_msg, created_at, processed_at
            FROM embedding_queue
            WHERE status = 'pending' AND retry_count < ?
            ORDER BY created_at
            LIMIT 1
        """, (MAX_RETRY,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return QueueItem(
                id=row[0],
                chunk_id=row[1],
                content=row[2],
                status=row[3],
                retry_count=row[4],
                error_msg=row[5],
                created_at=row[6],
                processed_at=row[7]
            )
        return None
    
    def reset_failed_items(self):
        """重置失败的项（retry_count < MAX_RETRY）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE embedding_queue 
            SET status = 'pending'
            WHERE status = 'failed' AND retry_count < ?
        """, (MAX_RETRY,))
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if count > 0:
            logger.info(f"重置 {count} 个失败项为 pending")
        return count
    
    def mark_processing(self, item: QueueItem):
        """标记为处理中"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE embedding_queue 
            SET status = 'processing', processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (item.id,))
        
        conn.commit()
        conn.close()
    
    def mark_done(self, item: QueueItem, vector_id: str):
        """标记完成"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 更新队列
        cursor.execute("""
            UPDATE embedding_queue 
            SET status = 'done', vector_id = ?, processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (vector_id, item.id))
        
        # 更新 chunks 表
        cursor.execute("""
            UPDATE chunks SET vector_id = ? WHERE id = ?
        """, (vector_id, item.chunk_id))
        
        conn.commit()
        conn.close()
    
    def mark_failed(self, item: QueueItem, error: str):
        """标记失败"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE embedding_queue 
            SET status = 'failed', retry_count = retry_count + 1, error_msg = ?, processed_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (error[:200], item.id))
        
        conn.commit()
        conn.close()
    
    def process_embedding(self, content: str) -> Optional[list]:
        """调用 Ollama embedding"""
        try:
            response = self.client.post(
                f"{OLLAMA_HOST}/api/embeddings",
                json={"model": OLLAMA_MODEL, "prompt": content},
                timeout=25.0
            )
            
            if response.status_code != 200:
                error_data = response.json()
                error = error_data.get("error", "未知错误")
                
                # 判断是否是输入过长（不重试）
                if "input length exceeds" in error.lower():
                    logger.warning(f"输入过长，跳过: {len(content)} 字符")
                    return None
                
                logger.warning(f"Ollama {response.status_code}: {error[:80]}")
                return None
            
            embedding = response.json().get("embedding")
            
            if not embedding:
                logger.warning("返回空 embedding")
                return None
            
            if all(v == 0.0 for v in embedding):
                logger.warning("返回零向量（服务未就绪）")
                return None
            
            return embedding
            
        except httpx.TimeoutException:
            logger.warning("Ollama 请求超时")
            return None
        except Exception as e:
            logger.error(f"请求错误: {e}")
            return None
    
    def get_stats(self) -> dict:
        """获取队列统计"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 队列统计
        cursor.execute("""
            SELECT status, COUNT(*) 
            FROM embedding_queue 
            GROUP BY status
        """)
        
        stats = {"pending": 0, "processing": 0, "done": 0, "failed": 0}
        for row in cursor.fetchall():
            stats[row[0]] = row[1]
        
        # chunks 统计
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE vector_id IS NULL")
        stats["chunks_no_vector"] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chunks WHERE vector_id IS NOT NULL")
        stats["chunks_with_vector"] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def run_worker(self, max_items: int = 0):
        """Worker 主循环
        
        Args:
            max_items: 最多处理多少项（0=无限制）
        """
        logger.info("=== Embedding Queue Worker 启动 ===")
        logger.info(f"Ollama: {OLLAMA_HOST}")
        logger.info(f"模型: {OLLAMA_MODEL}")
        logger.info(f"延迟: {PROCESS_DELAY}s")
        logger.info(f"最大重试: {MAX_RETRY}")
        
        # 填充队列
        self.populate_queue()
        
        # 重置失败的项
        self.reset_failed_items()
        
        stats = self.get_stats()
        logger.info(f"队列状态: pending={stats['pending']}, done={stats['done']}, failed={stats['failed']}")
        logger.info(f"Chunks 状态: 无向量={stats['chunks_no_vector']}, 有向量={stats['chunks_with_vector']}")
        
        processed = 0
        failed = 0
        skipped = 0
        
        while self._running:
            # 获取下一个
            item = self.get_pending_item()
            
            if not item:
                # 尝试填充新项
                new_items = self.populate_queue()
                
                if new_items == 0:
                    # 真的没有更多了
                    stats = self.get_stats()
                    if stats['pending'] == 0:
                        logger.info("队列已空")
                        break
                    
                    logger.info(f"等待新项... (pending={stats['pending']})")
                    time.sleep(5)
                    continue
            
            # 检查最大处理数
            if max_items > 0 and processed + failed >= max_items:
                logger.info(f"达到最大处理数 {max_items}")
                break
            
            # 标记处理中
            self.mark_processing(item)
            
            logger.info(f"处理 [{item.id}] chunk={item.chunk_id[:8]}... retry={item.retry_count}")
            
            # 获取 chunk 完整信息
            chunk_info = self.get_chunk_info(item.chunk_id)
            
            if not chunk_info:
                logger.warning(f"Chunk {item.chunk_id} 信息不存在")
                self.mark_failed(item, "chunk 信息不存在")
                failed += 1
                time.sleep(PROCESS_DELAY)
                continue
            
            # 截断保护
            content = item.content
            if len(content) > MAX_CHARS:
                content = content[:MAX_CHARS]
            
            # 向量化
            embedding = self.process_embedding(content)
            
            if embedding:
                # 成功：写入 Qdrant 向量库
                try:
                    vector_store = self._get_vector_store()
                    
                    # 解析 metadata
                    metadata = {}
                    if chunk_info.get("metadata_json"):
                        try:
                            metadata = json.loads(chunk_info["metadata_json"])
                        except:
                            pass
                    
                    payload = {
                        "chunk_id": item.chunk_id,
                        "document_id": chunk_info.get("document_id"),
                        "content": content,
                        "filename": chunk_info.get("filename", "unknown"),
                        "source_path": chunk_info.get("source_path"),
                        "start_line": metadata.get("start_line"),
                        "end_line": metadata.get("end_line"),
                    }
                    
                    # 确保集合存在
                    project_id = chunk_info.get("project_id")
                    if project_id:
                        vector_store.create_collection(project_id)
                    
                    # 添加到 Qdrant
                    vector_id = str(uuid.uuid4())
                    actual_id = vector_store.add_vector(
                        project_id=project_id,
                        vector=embedding,
                        payload=payload,
                        vector_id=vector_id
                    )
                    
                    if actual_id:
                        self.mark_done(item, actual_id)
                        processed += 1
                        
                        if processed % 20 == 0:
                            logger.info(f"进度: 成功 {processed}, 失败 {failed}, 跳过 {skipped}")
                    else:
                        logger.error("Qdrant 写入失败")
                        self.mark_failed(item, "Qdrant 写入失败")
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"向量写入错误: {e}")
                    self.mark_failed(item, str(e)[:200])
                    failed += 1
            else:
                # 向量化失败
                self.mark_failed(item, "向量化失败")
                failed += 1
            
            # 延迟保护
            time.sleep(PROCESS_DELAY)
        
        logger.info(f"=== 完成: 成功 {processed}, 失败 {failed}, 跳过 {skipped} ===")
        
        # 最终统计
        stats = self.get_stats()
        logger.info(f"队列状态: done={stats['done']}, pending={stats['pending']}, failed={stats['failed']}")
        logger.info(f"Chunks 状态: 无向量={stats['chunks_no_vector']}, 有向量={stats['chunks_with_vector']}")
        
        # 计算覆盖率
        total = stats['chunks_with_vector'] + stats['chunks_no_vector']
        if total > 0:
            coverage = stats['chunks_with_vector'] / total * 100
            logger.info(f"向量覆盖率: {coverage:.1f}%")
    
    def stop(self):
        """停止 worker"""
        self._running = False
        self.client.close()
        if self.vector_store:
            self.vector_store.client.close()
        logger.info("Worker 停止信号")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Queue Worker")
    parser.add_argument("--max", type=int, default=0, help="最多处理多少项（0=无限制）")
    parser.add_argument("--daemon", action="store_true", help="以守护进程模式运行（持续监控）")
    args = parser.parse_args()
    
    queue = EmbeddingQueue(DB_PATH)
    
    # 信号处理
    def handle_signal(sig, frame):
        logger.info(f"收到信号 {sig}")
        queue.stop()
    
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    
    # 运行
    if args.daemon:
        logger.info("守护进程模式启动")
        while queue._running:
            queue.run_worker(max_items=args.max)
            if queue._running:
                logger.info("等待新项... (30s)")
                time.sleep(30)
                queue.populate_queue()
    else:
        queue.run_worker(max_items=args.max)


if __name__ == "__main__":
    main()