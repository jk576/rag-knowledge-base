#!/usr/bin/env python3
"""Retry Failed Vectors

重试向量失败的 chunks。

用法:
    # 查看失败统计
    python scripts/retry_failed_vectors.py --stats
    
    # 查看失败详情
    python scripts/retry_failed_vectors.py --list --limit 20
    
    # 重试指定项目的失败向量
    python scripts/retry_failed_vectors.py --project 98bf56c9-xxx
    
    # 重试所有失败向量
    python scripts/retry_failed_vectors.py --all
    
    # 重试指定 chunks
    python scripts/retry_failed_vectors.py --chunks chunk_id1 chunk_id2
"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

import sqlite3
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("retry-vectors")

DB_PATH = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
MAX_RETRY = 3


def get_stats() -> dict:
    """获取向量状态统计"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 总体统计
    cursor.execute("""
        SELECT vector_status, COUNT(*) 
        FROM chunks 
        GROUP BY vector_status
    """)
    stats = dict(cursor.fetchall())
    
    # 按项目统计失败数
    cursor.execute("""
        SELECT p.name, COUNT(c.id) as failed_count
        FROM chunks c
        JOIN projects p ON c.project_id = p.id
        WHERE c.vector_status = 'failed'
        GROUP BY c.project_id
        ORDER BY failed_count DESC
    """)
    stats["by_project"] = cursor.fetchall()
    
    # 可重试数量
    cursor.execute("""
        SELECT COUNT(*) FROM chunks 
        WHERE vector_status = 'failed' 
        AND vector_retry_count < ?
    """, (MAX_RETRY,))
    stats["retryable"] = cursor.fetchone()[0]
    
    conn.close()
    return stats


def list_failed(project_id: str = None, limit: int = 50) -> list:
    """列出失败的 chunks"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if project_id:
        cursor.execute("""
            SELECT c.id, c.vector_error, c.vector_retry_count, 
                   c.last_vector_attempt, d.filename
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.project_id = ? AND c.vector_status = 'failed'
            ORDER BY c.last_vector_attempt DESC
            LIMIT ?
        """, (project_id, limit))
    else:
        cursor.execute("""
            SELECT c.id, c.vector_error, c.vector_retry_count, 
                   c.last_vector_attempt, d.filename, c.project_id
            FROM chunks c
            LEFT JOIN documents d ON c.document_id = d.id
            WHERE c.vector_status = 'failed'
            ORDER BY c.last_vector_attempt DESC
            LIMIT ?
        """, (limit,))
    
    results = cursor.fetchall()
    conn.close()
    return results


def reset_failed(project_id: str = None, chunk_ids: list = None) -> int:
    """重置失败的 chunks 为 pending"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(f"""
            UPDATE chunks 
            SET vector_status = 'pending', vector_error = NULL
            WHERE id IN ({placeholders})
            AND vector_status = 'failed'
        """, chunk_ids)
    elif project_id:
        cursor.execute("""
            UPDATE chunks 
            SET vector_status = 'pending', vector_error = NULL
            WHERE project_id = ? AND vector_status = 'failed'
        """, (project_id,))
    else:
        cursor.execute("""
            UPDATE chunks 
            SET vector_status = 'pending', vector_error = NULL
            WHERE vector_status = 'failed'
        """)
    
    count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return count


def enqueue_for_retry(project_id: str = None, chunk_ids: list = None) -> int:
    """将失败的 chunks 加入向量化队列"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 先重置状态
    reset_count = reset_failed(project_id, chunk_ids)
    
    if reset_count == 0:
        conn.close()
        return 0
    
    # 加入队列
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        cursor.execute(f"""
            INSERT OR IGNORE INTO embedding_queue (chunk_id, content)
            SELECT id, content FROM chunks 
            WHERE id IN ({placeholders})
            AND (vector_id IS NULL OR vector_id = '')
        """, chunk_ids)
    elif project_id:
        cursor.execute("""
            INSERT OR IGNORE INTO embedding_queue (chunk_id, content)
            SELECT id, content FROM chunks 
            WHERE project_id = ?
            AND (vector_id IS NULL OR vector_id = '')
        """, (project_id,))
    else:
        cursor.execute("""
            INSERT OR IGNORE INTO embedding_queue (chunk_id, content)
            SELECT id, content FROM chunks 
            WHERE vector_status = 'pending'
            AND (vector_id IS NULL OR vector_id = '')
        """)
    
    queued = cursor.rowcount
    conn.commit()
    conn.close()
    
    return queued


def main():
    parser = argparse.ArgumentParser(description="重试向量失败的 chunks")
    parser.add_argument("--stats", action="store_true", help="显示统计信息")
    parser.add_argument("--list", action="store_true", help="列出失败的 chunks")
    parser.add_argument("--project", type=str, help="指定项目 ID")
    parser.add_argument("--chunks", nargs="+", help="指定 chunk IDs")
    parser.add_argument("--all", action="store_true", help="重试所有失败项")
    parser.add_argument("--limit", type=int, default=50, help="限制返回数量")
    parser.add_argument("--dry-run", action="store_true", help="只显示不执行")
    args = parser.parse_args()
    
    if args.stats:
        stats = get_stats()
        print("\n=== 向量状态统计 ===")
        print(f"  success: {stats.get('success', 0)}")
        print(f"  pending: {stats.get('pending', 0)}")
        print(f"  failed:  {stats.get('failed', 0)}")
        print(f"  可重试: {stats.get('retryable', 0)} (retry_count < {MAX_RETRY})")
        
        if stats["by_project"]:
            print("\n=== 按项目统计失败数 ===")
            for name, count in stats["by_project"]:
                print(f"  {name}: {count}")
        return
    
    if args.list:
        results = list_failed(args.project, args.limit)
        print(f"\n=== 失败的 Chunks ({len(results)} 个) ===")
        for row in results:
            chunk_id, error, retry_count, last_attempt, filename = row[:5]
            project_id = row[5] if len(row) > 5 else args.project
            print(f"\n  ID: {chunk_id}")
            print(f"  文件: {filename}")
            if project_id:
                print(f"  项目: {project_id}")
            print(f"  错误: {error}")
            print(f"  重试次数: {retry_count}")
            print(f"  最后尝试: {last_attempt}")
        return
    
    if args.all or args.project or args.chunks:
        if args.dry_run:
            # 只显示将被重置的数量
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            if args.chunks:
                placeholders = ",".join("?" * len(args.chunks))
                cursor.execute(f"""
                    SELECT COUNT(*) FROM chunks 
                    WHERE id IN ({placeholders}) AND vector_status = 'failed'
                """, args.chunks)
            elif args.project:
                cursor.execute("""
                    SELECT COUNT(*) FROM chunks 
                    WHERE project_id = ? AND vector_status = 'failed'
                """, (args.project,))
            else:
                cursor.execute("""
                    SELECT COUNT(*) FROM chunks WHERE vector_status = 'failed'
                """)
            count = cursor.fetchone()[0]
            conn.close()
            print(f"[Dry Run] 将重置 {count} 个失败 chunks 为 pending")
            return
        
        # 执行重试
        count = reset_failed(args.project, args.chunks)
        queued = enqueue_for_retry(args.project, args.chunks)
        
        print(f"✅ 重置 {count} 个失败 chunks 为 pending")
        print(f"✅ 加入队列 {queued} 个 chunks")
        print("\n提示: 运行 embedding_queue_worker.py 来处理队列")
        return
    
    # 默认显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()