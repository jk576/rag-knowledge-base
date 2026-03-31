"""修复数据库中的超长 chunks

功能：
1. 找出所有超过 MAX_CHUNK_SIZE 的 chunks
2. 使用新的语义分块器重新分块
3. 更新数据库
4. 删除旧的向量（如果存在）
"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

import sqlite3
from pathlib import Path
from typing import List, Dict, Any
import uuid
from datetime import datetime

from src.core.semantic_chunker import SemanticChunker
from src.rag_api.config import get_settings

settings = get_settings()


def find_overlong_chunks(db_path: Path, max_size: int = None) -> List[Dict[str, Any]]:
    """找出所有超长 chunks
    
    Returns:
        超长 chunk 列表，每个包含 id, document_id, project_id, chunk_index, content, length
    """
    max_size = max_size or settings.MAX_CHUNK_SIZE
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, document_id, project_id, chunk_index, content, LENGTH(content) as len
        FROM chunks
        WHERE LENGTH(content) > ?
        ORDER BY LENGTH(content) DESC
    """, (max_size,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "document_id": row[1],
            "project_id": row[2],
            "chunk_index": row[3],
            "content": row[4],
            "length": row[5]
        }
        for row in rows
    ]


def rechunk_and_update(db_path: Path, overlong_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """重新分块并更新数据库
    
    Returns:
        统计信息：total_processed, total_new_chunks, errors
    """
    chunker = SemanticChunker()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {
        "total_processed": 0,
        "total_new_chunks": 0,
        "errors": [],
        "details": []
    }
    
    for old_chunk in overlong_chunks:
        try:
            # 使用新的分块器重新分块
            new_chunks = chunker.chunk_text(old_chunk["content"])
            
            if len(new_chunks) <= 1:
                # 如果只有一个 chunk，说明不需要分块（可能已经在合理范围内）
                # 只检查是否超过上限，如果超过就截断
                if len(new_chunks[0]) > settings.MAX_CHUNK_SIZE:
                    # 强制截断（最后手段）
                    truncated = new_chunks[0][:settings.MAX_CHUNK_SIZE]
                    cursor.execute(
                        "UPDATE chunks SET content = ? WHERE id = ?",
                        (truncated, old_chunk["id"])
                    )
                    stats["details"].append({
                        "old_id": old_chunk["id"],
                        "action": "truncated",
                        "old_len": old_chunk["length"],
                        "new_len": len(truncated)
                    })
                continue
            
            # 删除旧 chunk
            cursor.execute("DELETE FROM chunks WHERE id = ?", (old_chunk["id"],))
            
            # 创建新 chunks（保留 project_id 和 chunk_index）
            for i, new_content in enumerate(new_chunks):
                new_id = str(uuid.uuid4())
                # chunk_index 使用原来的值 + 偏移，保持顺序
                new_chunk_index = old_chunk["chunk_index"] + i
                cursor.execute("""
                    INSERT INTO chunks (id, document_id, project_id, chunk_index, content, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    new_id, 
                    old_chunk["document_id"], 
                    old_chunk["project_id"],
                    new_chunk_index,
                    new_content, 
                    datetime.now().isoformat()
                ))
                
                stats["total_new_chunks"] += 1
            
            stats["total_processed"] += 1
            stats["details"].append({
                "old_id": old_chunk["id"],
                "action": "rechunked",
                "old_len": old_chunk["length"],
                "new_chunks": len(new_chunks),
                "new_lens": [len(c) for c in new_chunks]
            })
            
        except Exception as e:
            stats["errors"].append({
                "chunk_id": old_chunk["id"],
                "error": str(e)
            })
    
    conn.commit()
    conn.close()
    
    return stats


def verify_fix(db_path: Path, max_size: int = None) -> Dict[str, Any]:
    """验证修复结果
    
    Returns:
        验证信息：remaining_overlong, total_chunks, max_len
    """
    max_size = max_size or settings.MAX_CHUNK_SIZE
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 统计总 chunk 数
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    
    # 检查是否还有超长 chunk
    cursor.execute("""
        SELECT COUNT(*), MAX(LENGTH(content))
        FROM chunks
        WHERE LENGTH(content) > ?
    """, (max_size,))
    
    remaining_overlong = cursor.fetchone()[0]
    max_len_result = cursor.execute("SELECT MAX(LENGTH(content)) FROM chunks").fetchone()[0]
    
    conn.close()
    
    return {
        "total_chunks": total_chunks,
        "remaining_overlong": remaining_overlong,
        "max_len": max_len_result,
        "passed": remaining_overlong == 0
    }


def main():
    """主函数"""
    db_path = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
    
    if not db_path.exists():
        print("❌ 数据库不存在")
        return
    
    print("=== 修复超长 chunks ===")
    print(f"MAX_CHUNK_SIZE: {settings.MAX_CHUNK_SIZE} (调整后)")
    print()
    
    # Step 1: 找出超长 chunks
    print("Step 1: 查找超长 chunks...")
    overlong_chunks = find_overlong_chunks(db_path)
    
    if not overlong_chunks:
        print("✅ 没有发现超长 chunks")
        return
    
    print(f"发现 {len(overlong_chunks)} 个超长 chunks")
    print(f"最长: {overlong_chunks[0]['length']} 字符")
    print()
    
    # 显示前 5 个详情
    print("超长 chunks 详情（前5个）:")
    for i, chunk in enumerate(overlong_chunks[:5]):
        print(f"  {i+1}. ID={chunk['id'][:8]}... project={chunk['project_id'][:8]}... len={chunk['length']}")
    print()
    
    # Step 2: 重新分块
    print("Step 2: 重新分块...")
    stats = rechunk_and_update(db_path, overlong_chunks)
    
    print(f"处理完成: {stats['total_processed']} 个超长 chunks")
    print(f"创建新 chunks: {stats['total_new_chunks']} 个")
    
    if stats['errors']:
        print(f"错误: {len(stats['errors'])} 个")
        for err in stats['errors'][:5]:
            print(f"  - {err['chunk_id'][:8]}...: {err['error']}")
    print()
    
    # Step 3: 验证
    print("Step 3: 验证修复结果...")
    verify_result = verify_fix(db_path)
    
    print(f"总 chunks 数: {verify_result['total_chunks']}")
    print(f"剩余超长 chunks: {verify_result['remaining_overlong']}")
    print(f"当前最大长度: {verify_result['max_len']}")
    
    if verify_result['passed']:
        print("✅ 所有 chunks 符合上限要求")
    else:
        print("❌ 仍有超长 chunks，需要进一步处理")
    
    # 显示详情
    if stats['details']:
        print()
        print("=== 处理详情 ===")
        for detail in stats['details'][:10]:
            if detail['action'] == 'rechunked':
                print(f"  {detail['old_id'][:8]}...: {detail['old_len']} → {detail['new_chunks']} chunks")
                print(f"    新 chunks 长度: {detail['new_lens'][:5]}...")
    
    print()
    print("下一步: 运行 revectorize_failed_chunks.py")


if __name__ == "__main__":
    main()