"""Embedding Queue Manager

提供给 DocumentService 使用的队列管理接口。
新文档入库时，将 chunks 写入队列而非直接向量化。
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class EmbeddingQueueManager:
    """Embedding 队列管理器
    
    使用方式：
    1. DocumentService 创建 chunks 后调用 queue_chunks()
    2. Queue Worker 后台处理队列
    3. 通过 get_queue_status() 查看进度
    """
    
    def __init__(self, db_path: Path, max_chars: int = 4000):
        self.db_path = db_path
        self.max_chars = max_chars
        self._ensure_table()
    
    def _ensure_table(self):
        """确保队列表存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='embedding_queue'
        """)
        
        if not cursor.fetchone():
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
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_queue_status 
                ON embedding_queue(status, retry_count)
            """)
            
            conn.commit()
            logger.info("创建 embedding_queue 表")
        
        conn.close()
    
    def queue_chunks(
        self,
        chunks: List[Dict[str, Any]],
        skip_existing: bool = True
    ) -> Dict[str, int]:
        """将 chunks 加入向量化队列
        
        Args:
            chunks: [{"id": str, "content": str, "document_id": str, "project_id": str}, ...]
            skip_existing: 是否跳过已存在的
            
        Returns:
            {"queued": int, "skipped": int, "total": int}
        """
        if not chunks:
            return {"queued": 0, "skipped": 0, "total": 0}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        queued = 0
        skipped = 0
        
        for chunk in chunks:
            chunk_id = chunk.get("id")
            content = chunk.get("content", "")
            
            if not chunk_id:
                skipped += 1
                continue
            
            # 截断保护
            if len(content) > self.max_chars:
                content = content[:self.max_chars]
            
            try:
                if skip_existing:
                    cursor.execute("""
                        INSERT OR IGNORE INTO embedding_queue (chunk_id, content)
                        VALUES (?, ?)
                    """, (chunk_id, content))
                else:
                    cursor.execute("""
                        INSERT OR REPLACE INTO embedding_queue (chunk_id, content, status)
                        VALUES (?, ?, 'pending')
                    """, (chunk_id, content))
                
                if cursor.rowcount > 0:
                    queued += 1
                else:
                    skipped += 1
                    
            except sqlite3.IntegrityError:
                skipped += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"队列: 加入 {queued} 个, 跳过 {skipped} 个")
        
        return {
            "queued": queued,
            "skipped": skipped,
            "total": len(chunks)
        }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """获取队列状态
        
        Returns:
            {
                "pending": int,
                "processing": int,
                "done": int,
                "failed": int,
                "chunks_no_vector": int,
                "chunks_with_vector": int,
                "coverage": float
            }
        """
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
        
        # 计算覆盖率
        total = stats["chunks_with_vector"] + stats["chunks_no_vector"]
        stats["coverage"] = stats["chunks_with_vector"] / total * 100 if total > 0 else 0
        
        conn.close()
        return stats
    
    def clear_done_items(self, days: int = 7) -> int:
        """清理已完成的旧项
        
        Args:
            days: 保留天数
            
        Returns:
            清理数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM embedding_queue 
            WHERE status = 'done' 
            AND processed_at < datetime('now', ?)
        """, (f'-{days} days',))
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if count > 0:
            logger.info(f"清理 {count} 个已完成项（>{days}天）")
        
        return count
    
    def reset_failed_items(self, max_retry: int = 3) -> int:
        """重置失败项（未达到最大重试次数）
        
        Args:
            max_retry: 最大重试次数
            
        Returns:
            重置数量
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE embedding_queue 
            SET status = 'pending'
            WHERE status = 'failed' AND retry_count < ?
        """, (max_retry,))
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        
        if count > 0:
            logger.info(f"重置 {count} 个失败项为 pending")
        
        return count
    
    def get_failed_items(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取失败项详情
        
        Args:
            limit: 最大返回数
            
        Returns:
            [{"chunk_id": str, "error_msg": str, "retry_count": int}, ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT chunk_id, error_msg, retry_count, created_at
            FROM embedding_queue
            WHERE status = 'failed'
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        items = []
        for row in cursor.fetchall():
            items.append({
                "chunk_id": row[0],
                "error_msg": row[1],
                "retry_count": row[2],
                "created_at": row[3]
            })
        
        conn.close()
        return items


# 全局实例
_queue_manager = None


def get_queue_manager(db_path: Optional[Path] = None) -> EmbeddingQueueManager:
    """获取全局队列管理器实例"""
    global _queue_manager
    
    if _queue_manager is None:
        from src.rag_api.config import get_settings
        settings = get_settings()
        
        if db_path is None:
            # 使用正确的数据库路径
            db_path = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
        
        _queue_manager = EmbeddingQueueManager(db_path)
    
    return _queue_manager