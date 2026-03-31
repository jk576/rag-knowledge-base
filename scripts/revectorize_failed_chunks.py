"""重新处理向量化失败的 chunks

功能：
1. 找出没有 vector_id 的 chunks
2. 检查 Ollama 服务状态
3. 使用新的 retry 机制重新向量化
4. 更新 Qdrant 向量库
"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import time

from src.core.embedding import EmbeddingService
from src.core.vector_store import VectorStore
from src.rag_api.config import get_settings

settings = get_settings()


def find_chunks_without_vectors(db_path: Path, project_id: str = None) -> List[Dict[str, Any]]:
    """找出没有向量的 chunks
    
    Args:
        db_path: 数据库路径
        project_id: 项目 ID（可选，用于筛选特定项目）
    
    Returns:
        没有 vector_id 的 chunks 列表
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if project_id:
        cursor.execute("""
            SELECT id, document_id, project_id, chunk_index, content, metadata_json
            FROM chunks
            WHERE project_id = ? AND vector_id IS NULL
            ORDER BY chunk_index
        """, (project_id,))
    else:
        cursor.execute("""
            SELECT id, document_id, project_id, chunk_index, content, metadata_json
            FROM chunks
            WHERE vector_id IS NULL
            ORDER BY project_id, chunk_index
        """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "document_id": row[1],
            "project_id": row[2],
            "chunk_index": row[3],
            "content": row[4],
            "metadata_json": row[5]
        }
        for row in rows
    ]


def revectorize_chunks(
    chunks: List[Dict[str, Any]], 
    embedding_service: EmbeddingService,
    vector_store: VectorStore,
    batch_size: int = 10,
    delay_ms: int = 100
) -> Dict[str, Any]:
    """重新向量化 chunks
    
    Args:
        chunks: 待处理的 chunks
        embedding_service: Embedding 服务
        vector_store: 向量存储服务
        batch_size: 每批次处理数量
        delay_ms: 每批次之间的延迟（毫秒）
    
    Returns:
        统计信息：success_count, failed_count, errors
    """
    stats = {
        "total": len(chunks),
        "success": 0,
        "failed": 0,
        "errors": [],
        "zero_vectors": 0
    }
    
    # 按项目分组
    by_project: Dict[str, List[Dict[str, Any]]] = {}
    for chunk in chunks:
        pid = chunk["project_id"]
        if pid not in by_project:
            by_project[pid] = []
        by_project[pid].append(chunk)
    
    # 分批处理
    for project_id, project_chunks in by_project.items():
        print(f"\n处理项目 {project_id[:8]}... ({len(project_chunks)} chunks)")
        
        for i in range(0, len(project_chunks), batch_size):
            batch = project_chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"  批次 {batch_num}/{(len(project_chunks) // batch_size) + 1}...")
            
            # 向量化
            embeddings = []
            valid_chunks = []
            
            for chunk in batch:
                try:
                    # 使用新的 retry 机制
                    emb = embedding_service.embed_text_sync(chunk["content"])
                    
                    # 检查是否是零向量（向量化失败）
                    if all(v == 0.0 for v in emb):
                        stats["zero_vectors"] += 1
                        stats["errors"].append({
                            "chunk_id": chunk["id"],
                            "error": "零向量（向量化失败）",
                            "content_len": len(chunk["content"])
                        })
                        continue
                    
                    embeddings.append(emb)
                    valid_chunks.append(chunk)
                    
                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append({
                        "chunk_id": chunk["id"],
                        "error": str(e)[:100],
                        "content_len": len(chunk["content"])
                    })
            
            # 添加到 Qdrant
            if embeddings:
                try:
                    # 构建 payloads
                    payloads = []
                    for chunk in valid_chunks:
                        metadata = json.loads(chunk["metadata_json"]) if chunk["metadata_json"] else {}
                        payloads.append({
                            "chunk_id": chunk["id"],
                            "document_id": chunk["document_id"],
                            "content": chunk["content"],
                            "start_line": metadata.get("start_line"),
                            "end_line": metadata.get("end_line"),
                        })
                    
                    # 添加向量
                    vector_ids = vector_store.add_vectors_batch(
                        project_id=project_id,
                        vectors=embeddings,
                        payloads=payloads
                    )
                    
                    # 更新数据库
                    db_path = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    for chunk, vector_id in zip(valid_chunks, vector_ids):
                        cursor.execute(
                            "UPDATE chunks SET vector_id = ? WHERE id = ?",
                            (vector_id, chunk["id"])
                        )
                    
                    conn.commit()
                    conn.close()
                    
                    stats["success"] += len(valid_chunks)
                    print(f"    ✅ 成功: {len(valid_chunks)}/{len(batch)}")
                    
                except Exception as e:
                    stats["failed"] += len(valid_chunks)
                    print(f"    ❌ Qdrant 错误: {e}")
                    for chunk in valid_chunks:
                        stats["errors"].append({
                            "chunk_id": chunk["id"],
                            "error": f"Qdrant: {str(e)[:100]}"
                        })
            
            # 延迟（防止 Ollama 过载）
            if delay_ms > 0 and i + batch_size < len(project_chunks):
                time.sleep(delay_ms / 1000)
    
    return stats


def check_ollama_status(embedding_service: EmbeddingService) -> bool:
    """检查 Ollama 服务状态"""
    try:
        import httpx
        response = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5.0)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("name") for m in data.get("models", [])]
            print(f"✅ Ollama 运行正常")
            print(f"   可用模型: {models}")
            return True
    except Exception as e:
        print(f"❌ Ollama 服务不可用: {e}")
        return False


def get_statistics(db_path: Path) -> Dict[str, Any]:
    """获取数据库统计信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 总 chunks 数
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total_chunks = cursor.fetchone()[0]
    
    # 有向量的 chunks
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE vector_id IS NOT NULL")
    with_vectors = cursor.fetchone()[0]
    
    # 无向量的 chunks
    without_vectors = total_chunks - with_vectors
    
    # 按项目统计
    cursor.execute("""
        SELECT project_id, 
               COUNT(*) as total,
               SUM(CASE WHEN vector_id IS NULL THEN 1 ELSE 0 END) as no_vector
        FROM chunks
        GROUP BY project_id
    """)
    project_stats = cursor.fetchall()
    
    conn.close()
    
    return {
        "total_chunks": total_chunks,
        "with_vectors": with_vectors,
        "without_vectors": without_vectors,
        "coverage": with_vectors / total_chunks * 100 if total_chunks > 0 else 0,
        "projects": [
            {
                "project_id": row[0],
                "total": row[1],
                "no_vector": row[2],
                "coverage": (row[1] - row[2]) / row[1] * 100 if row[1] > 0 else 0
            }
            for row in project_stats
        ]
    }


def main():
    """主函数"""
    db_path = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
    
    if not db_path.exists():
        print("❌ 数据库不存在")
        return
    
    print("=== 重新处理向量化失败的 chunks ===")
    print()
    
    # Step 1: 检查 Ollama
    print("Step 1: 检查 Ollama 服务...")
    embedding_service = EmbeddingService()
    
    if not check_ollama_status(embedding_service):
        print("请先启动 Ollama 服务: ollama serve")
        return
    
    print()
    
    # Step 2: 统计信息
    print("Step 2: 当前状态...")
    stats = get_statistics(db_path)
    
    print(f"总 chunks: {stats['total_chunks']}")
    print(f"有向量: {stats['with_vectors']} ({stats['coverage']:.1f}%)")
    print(f"无向量: {stats['without_vectors']}")
    print()
    
    print("项目统计:")
    for proj in stats['projects']:
        print(f"  {proj['project_id'][:8]}...: {proj['total']} chunks, {proj['no_vector']} 无向量 ({proj['coverage']:.1f}%)")
    print()
    
    # Step 3: 找出无向量的 chunks
    print("Step 3: 找出无向量的 chunks...")
    chunks = find_chunks_without_vectors(db_path)
    
    if not chunks:
        print("✅ 所有 chunks 都已有向量")
        return
    
    print(f"发现 {len(chunks)} 个 chunks 需要处理")
    print()
    
    # Step 4: 重新向量化
    print("Step 4: 重新向量化...")
    vector_store = VectorStore()
    
    result = revectorize_chunks(
        chunks,
        embedding_service,
        vector_store,
        batch_size=5,  # 小批次防止过载
        delay_ms=200   # 增加延迟
    )
    
    print()
    print("=== 处理结果 ===")
    print(f"总数: {result['total']}")
    print(f"成功: {result['success']}")
    print(f"失败: {result['failed']}")
    print(f"零向量: {result['zero_vectors']}")
    
    if result['errors']:
        print(f"\n错误详情（前10个）:")
        for err in result['errors'][:10]:
            print(f"  {err['chunk_id'][:8]}...: {err['error']} (len={err['content_len']})")
    
    # Step 5: 最终统计
    print()
    print("=== 最终状态 ===")
    final_stats = get_statistics(db_path)
    print(f"总 chunks: {final_stats['total_chunks']}")
    print(f"有向量: {final_stats['with_vectors']} ({final_stats['coverage']:.1f}%)")
    print(f"覆盖率提升: {final_stats['coverage'] - stats['coverage']:.1f}%")


if __name__ == "__main__":
    main()