"""批量向量化无向量的 chunks

简化版本，逐个处理并实时反馈
"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

import sqlite3
import json
import httpx
from pathlib import Path
from src.core.vector_store import VectorStore
from src.rag_api.config import get_settings
import time

settings = get_settings()

def main():
    db_path = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 获取总数
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE vector_id IS NULL")
    remaining = cursor.fetchone()[0]
    
    if remaining == 0:
        print("✅ 所有 chunks 都已有向量")
        conn.close()
        return
    
    print(f"待处理 chunks: {remaining}")
    print()
    
    client = httpx.Client(timeout=30.0)
    vector_store = VectorStore()
    
    batch_size = 20
    processed = 0
    success = 0
    failed = 0
    
    while remaining > 0:
        # 获取一批
        cursor.execute("""
            SELECT id, document_id, project_id, content, metadata_json
            FROM chunks
            WHERE vector_id IS NULL
            LIMIT ?
        """, (batch_size,))
        
        batch = cursor.fetchall()
        
        if not batch:
            break
        
        for chunk_id, doc_id, proj_id, content, metadata_json in batch:
            processed += 1
            
            try:
                # 截断保护
                if len(content) > settings.MAX_CHUNK_SIZE:
                    content = content[:settings.MAX_CHUNK_SIZE]
                
                # 向量化
                response = client.post(
                    f"{settings.OLLAMA_HOST}/api/embeddings",
                    json={"model": settings.OLLAMA_MODEL, "prompt": content}
                )
                
                if response.status_code != 200:
                    failed += 1
                    continue
                
                embedding = response.json().get("embedding")
                
                if not embedding or all(v == 0.0 for v in embedding):
                    failed += 1
                    continue
                
                # 添加向量
                metadata = json.loads(metadata_json) if metadata_json else {}
                payload = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "content": content,
                }
                
                vector_id = vector_store.add_vector(proj_id, embedding, payload)
                
                # 更新数据库
                cursor.execute(
                    "UPDATE chunks SET vector_id = ? WHERE id = ?",
                    (vector_id, chunk_id)
                )
                conn.commit()
                
                success += 1
                
                # 延迟
                time.sleep(0.1)
                
            except Exception as e:
                failed += 1
        
        # 进度报告
        remaining -= len(batch)
        print(f"进度: {processed}/{remaining+processed}, 成功: {success}, 失败: {failed}")
        
        # 每处理 200 个暂停一会
        if processed % 200 == 0:
            print("暂停 5 秒...")
            time.sleep(5)
    
    print()
    print("=== 完成 ===")
    print(f"总处理: {processed}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    
    client.close()
    conn.close()

if __name__ == "__main__":
    main()