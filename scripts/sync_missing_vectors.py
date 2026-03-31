"""Sync Missing Vectors to Qdrant

同步 SQLite 中有 vector_id 但 Qdrant 中缺失的向量。

问题根源：
- 早期批量写入时 vector_id 先写入 SQLite
- Qdrant upsert 可能部分失败
- 导致 SQLite 有 vector_id，但 Qdrant 没有实际向量

解决方案：
1. 对比 SQLite chunks 和 Qdrant vectors
2. 找出缺失的向量
3. 重新向量化并写入 Qdrant
"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

import sqlite3
import json
import httpx
import time
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("sync-vectors")

# 配置
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "bge-m3"
MAX_CHARS = 4000
PROCESS_DELAY = 0.2
DB_PATH = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")


def get_qdrant_client():
    """获取 Qdrant 客户端"""
    from qdrant_client import QdrantClient
    return QdrantClient(host='localhost', port=6333, check_compatibility=False)


def get_missing_chunks(project_id: str) -> List[Dict[str, Any]]:
    """获取项目中有 vector_id 但可能缺失 Qdrant 向量的 chunks"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, content, document_id, vector_id, metadata_json
        FROM chunks 
        WHERE project_id = ? AND vector_id IS NOT NULL
    """, (project_id,))
    
    chunks = []
    for row in cursor.fetchall():
        chunks.append({
            "id": row[0],
            "content": row[1],
            "document_id": row[2],
            "vector_id": row[3],
            "metadata_json": row[4]
        })
    
    conn.close()
    return chunks


def check_vectors_in_qdrant(project_id: str, chunk_ids: List[str]) -> set:
    """检查哪些 chunk_id 的向量在 Qdrant 中存在"""
    from qdrant_client.http import models
    
    client = get_qdrant_client()
    collection_name = f"project_{project_id}"
    
    # 获取所有 payload 中的 chunk_id
    existing_chunk_ids = set()
    
    try:
        # 使用 scroll 获取所有点
        result = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        for point in result[0]:
            if point.payload and "chunk_id" in point.payload:
                existing_chunk_ids.add(point.payload["chunk_id"])
        
        # 如果第一批不够，继续获取
        while result[1]:  # 有 next_page_offset
            result = client.scroll(
                collection_name=collection_name,
                offset=result[1],
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            for point in result[0]:
                if point.payload and "chunk_id" in point.payload:
                    existing_chunk_ids.add(point.payload["chunk_id"])
    
    except Exception as e:
        logger.error(f"获取 Qdrant vectors 失败: {e}")
    
    return existing_chunk_ids


def embed_content(content: str, client: httpx.Client) -> List[float]:
    """调用 Ollama embedding"""
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS]
    
    try:
        response = client.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": OLLAMA_MODEL, "prompt": content},
            timeout=25.0
        )
        
        if response.status_code != 200:
            return []
        
        embedding = response.json().get("embedding")
        if not embedding or all(v == 0.0 for v in embedding):
            return []
        
        return embedding
    
    except Exception as e:
        logger.error(f"Embedding 错误: {e}")
        return []


def write_to_qdrant(project_id: str, chunk: Dict[str, Any], embedding: List[float]) -> bool:
    """写入向量到 Qdrant"""
    from src.core.vector_store import VectorStore
    
    vs = VectorStore()
    
    metadata = {}
    if chunk.get("metadata_json"):
        try:
            metadata = json.loads(chunk["metadata_json"])
        except:
            pass
    
    payload = {
        "chunk_id": chunk["id"],
        "document_id": chunk["document_id"],
        "content": chunk["content"][:MAX_CHARS],
        "start_line": metadata.get("start_line"),
        "end_line": metadata.get("end_line"),
    }
    
    try:
        # 确保集合存在
        vs.create_collection(project_id)
        
        # 写入向量
        result = vs.add_vector(
            project_id=project_id,
            vector=embedding,
            payload=payload
        )
        
        return result is not None
    
    except Exception as e:
        logger.error(f"Qdrant 写入错误: {e}")
        return False


def sync_project(project_id: str) -> Dict[str, int]:
    """同步单个项目的缺失向量"""
    logger.info(f"检查项目: {project_id}")
    
    # 1. 获取 SQLite chunks
    chunks = get_missing_chunks(project_id)
    sqlite_count = len(chunks)
    logger.info(f"SQLite 有 vector_id 的 chunks: {sqlite_count}")
    
    if sqlite_count == 0:
        return {"sqlite": 0, "qdrant": 0, "missing": 0, "synced": 0}
    
    # 2. 获取 Qdrant 已存在的 chunk_ids
    existing_ids = check_vectors_in_qdrant(project_id, [c["id"] for c in chunks])
    qdrant_count = len(existing_ids)
    logger.info(f"Qdrant 已存在向量: {qdrant_count}")
    
    # 3. 找出缺失的
    missing_chunks = [c for c in chunks if c["id"] not in existing_ids]
    missing_count = len(missing_chunks)
    logger.info(f"缺失向量: {missing_count}")
    
    if missing_count == 0:
        return {"sqlite": sqlite_count, "qdrant": qdrant_count, "missing": 0, "synced": 0}
    
    # 4. 同步缺失的向量
    synced = 0
    failed = 0
    
    httpx_client = httpx.Client(timeout=30.0)
    
    for i, chunk in enumerate(missing_chunks):
        logger.info(f"处理 [{i+1}/{missing_count}] chunk={chunk['id'][:8]}...")
        
        # 向量化
        embedding = embed_content(chunk["content"], httpx_client)
        
        if not embedding:
            logger.warning(f"向量化失败: {chunk['id'][:8]}")
            failed += 1
            time.sleep(PROCESS_DELAY)
            continue
        
        # 写入 Qdrant
        if write_to_qdrant(project_id, chunk, embedding):
            synced += 1
            if synced % 20 == 0:
                logger.info(f"进度: 同步 {synced}, 失败 {failed}")
        else:
            failed += 1
        
        # 延迟保护
        time.sleep(PROCESS_DELAY)
    
    httpx_client.close()
    
    logger.info(f"项目 {project_id} 完成: 同步 {synced}, 失败 {failed}")
    
    return {
        "sqlite": sqlite_count,
        "qdrant": qdrant,
        "missing": missing_count,
        "synced": synced,
        "failed": failed
    }


def fix_orphan_vectors(project_id: str, dry_run: bool = True) -> int:
    """
    清理孤儿向量（Qdrant 有但 SQLite 无）
    
    Args:
        project_id: 项目 ID
        dry_run: 只报告不执行
        
    Returns:
        清理的向量数
    """
    from qdrant_client.http import models
    
    client = get_qdrant_client()
    collection_name = f"project_{project_id}"
    
    # 1. 获取 SQLite 中所有 chunk_ids
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id FROM chunks WHERE project_id = ?
    """, (project_id,))
    sqlite_chunk_ids = set(row[0] for row in cursor.fetchall())
    conn.close()
    
    # 2. 获取 Qdrant 中所有向量
    qdrant_chunk_ids = set()
    qdrant_vector_ids = {}  # chunk_id -> vector_id
    
    try:
        result = client.scroll(
            collection_name=collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        
        for point in result[0]:
            if point.payload and "chunk_id" in point.payload:
                chunk_id = point.payload["chunk_id"]
                qdrant_chunk_ids.add(chunk_id)
                qdrant_vector_ids[chunk_id] = str(point.id)
        
        while result[1]:
            result = client.scroll(
                collection_name=collection_name,
                offset=result[1],
                limit=10000,
                with_payload=True,
                with_vectors=False
            )
            for point in result[0]:
                if point.payload and "chunk_id" in point.payload:
                    chunk_id = point.payload["chunk_id"]
                    qdrant_chunk_ids.add(chunk_id)
                    qdrant_vector_ids[chunk_id] = str(point.id)
    
    except Exception as e:
        logger.error(f"获取 Qdrant vectors 失败: {e}")
        return 0
    
    # 3. 找出孤儿向量
    orphan_chunk_ids = qdrant_chunk_ids - sqlite_chunk_ids
    
    if not orphan_chunk_ids:
        logger.info(f"项目 {project_id}: 无孤儿向量")
        return 0
    
    logger.info(f"项目 {project_id}: 发现 {len(orphan_chunk_ids)} 个孤儿向量")
    
    if dry_run:
        logger.info(f"[Dry Run] 将删除 {len(orphan_chunk_ids)} 个孤儿向量")
        for chunk_id in list(orphan_chunk_ids)[:10]:
            vector_id = qdrant_vector_ids.get(chunk_id)
            logger.info(f"  chunk_id={chunk_id}, vector_id={vector_id}")
        return len(orphan_chunk_ids)
    
    # 4. 执行删除
    orphan_vector_ids = [qdrant_vector_ids[cid] for cid in orphan_chunk_ids if cid in qdrant_vector_ids]
    
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=orphan_vector_ids)
        )
        logger.info(f"✅ 已删除 {len(orphan_vector_ids)} 个孤儿向量")
        return len(orphan_vector_ids)
        
    except Exception as e:
        logger.error(f"删除孤儿向量失败: {e}")
        return 0


def check_consistency(project_id: str) -> dict:
    """
    检查 SQLite 和 Qdrant 一致性
    
    Returns:
        {
            "sqlite_count": int,
            "qdrant_count": int,
            "missing_in_qdrant": int,
            "orphan_in_qdrant": int,
            "action_needed": str
        }
    """
    client = get_qdrant_client()
    collection_name = f"project_{project_id}"
    
    # 1. SQLite 统计
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*) FROM chunks WHERE project_id = ?
    """, (project_id,))
    sqlite_total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT COUNT(*) FROM chunks 
        WHERE project_id = ? AND vector_id IS NOT NULL
    """, (project_id,))
    sqlite_with_vector = cursor.fetchone()[0]
    
    conn.close()
    
    # 2. Qdrant 统计
    try:
        info = client.get_collection(collection_name)
        qdrant_count = info.points_count
    except:
        qdrant_count = 0
    
    # 3. 判断差异
    missing_in_qdrant = max(0, sqlite_with_vector - qdrant_count)
    orphan_in_qdrant = max(0, qdrant_count - sqlite_with_vector)
    
    if missing_in_qdrant > 0:
        action = "sync_missing"
    elif orphan_in_qdrant > 0:
        action = "clean_orphans"
    else:
        action = "none"
    
    return {
        "project_id": project_id,
        "sqlite_total": sqlite_total,
        "sqlite_with_vector": sqlite_with_vector,
        "qdrant_count": qdrant_count,
        "missing_in_qdrant": missing_in_qdrant,
        "orphan_in_qdrant": orphan_in_qdrant,
        "action_needed": action
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="同步向量到 Qdrant")
    parser.add_argument("--check", action="store_true", help="只检查一致性，不执行同步")
    parser.add_argument("--clean-orphans", action="store_true", help="清理孤儿向量")
    parser.add_argument("--project", type=str, help="指定项目 ID")
    parser.add_argument("--dry-run", action="store_true", help="只报告不执行")
    args = parser.parse_args()
    
    logger.info("=== 向量同步工具 ===")
    
    if args.project:
        # 单项目模式
        if args.check:
            result = check_consistency(args.project)
            print(f"\n项目 {args.project}:")
            print(f"  SQLite 总 chunks: {result['sqlite_total']}")
            print(f"  SQLite 有向量: {result['sqlite_with_vector']}")
            print(f"  Qdrant 向量数: {result['qdrant_count']}")
            print(f"  Qdrant 缺失: {result['missing_in_qdrant']}")
            print(f"  Qdrant 孤儿: {result['orphan_in_qdrant']}")
            print(f"  建议操作: {result['action_needed']}")
            return
        
        if args.clean_orphans:
            count = fix_orphan_vectors(args.project, dry_run=args.dry_run)
            if args.dry_run:
                print(f"[Dry Run] 将清理 {count} 个孤儿向量")
            else:
                print(f"✅ 已清理 {count} 个孤儿向量")
            return
        
        # 同步缺失向量
        result = sync_project(args.project)
        print(f"\n项目 {args.project} 同步完成:")
        print(f"  SQLite: {result['sqlite']}")
        print(f"  Qdrant: {result['qdrant']}")
        print(f"  缺失: {result['missing']}")
        print(f"  同步: {result['synced']}")
        print(f"  失败: {result.get('failed', 0)}")
        return
    
    # 全项目模式
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT project_id FROM chunks")
    projects = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    logger.info(f"共有 {len(projects)} 个项目")
    
    # 统计
    total_missing = 0
    total_synced = 0
    total_orphans = 0
    
    client = get_qdrant_client()
    
    for project_id in projects:
        result = check_consistency(project_id)
        
        if result['missing_in_qdrant'] > 0 and not args.check:
            logger.info(f"项目 {project_id}: 缺失 {result['missing_in_qdrant']} 个向量")
            sync_result = sync_project(project_id)
            total_missing += result['missing_in_qdrant']
            total_synced += sync_result['synced']
        
        if result['orphan_in_qdrant'] > 0:
            logger.info(f"项目 {project_id}: 发现 {result['orphan_in_qdrant']} 个孤儿向量")
            if args.clean_orphans and not args.dry_run and not args.check:
                count = fix_orphan_vectors(project_id, dry_run=False)
                total_orphans += count
    
    print(f"\n=== 总计 ===")
    print(f"缺失向量: {total_missing}")
    print(f"已同步: {total_synced}")
    print(f"孤儿清理: {total_orphans}")
    
    # 最终验证
    final_total = 0
    for c in client.get_collections().collections:
        info = client.get_collection(c.name)
        final_total += info.points_count
    
    print(f"Qdrant 总向量数: {final_total}")


if __name__ == "__main__":
    main()