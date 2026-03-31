#!/usr/bin/env python3
"""Cron Vector Sync

定期向量同步任务。

建议 crontab 配置:
    # 每日凌晨 3 点运行
    0 3 * * * /Users/jk/Projects/rag-knowledge-base/scripts/cron_vector_sync.py >> /var/log/rag_sync.log 2>&1

任务：
1. 检查所有项目的向量一致性
2. 同步缺失的向量
3. 清理孤儿向量
4. 重试失败的向量
5. 生成报告
"""

import sys
sys.path.insert(0, '/Users/jk/Projects/rag-knowledge-base')

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("cron-sync")

DB_PATH = Path("/Users/jk/Projects/rag-knowledge-base/db/metadata.db")
REPORT_DIR = Path("/Users/jk/Projects/rag-knowledge-base/reports")
MAX_RETRY = 3


def get_all_projects() -> List[Dict[str, Any]]:
    """获取所有项目"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, name, document_count, chunk_count
        FROM projects
        ORDER BY name
    """)
    
    projects = []
    for row in cursor.fetchall():
        projects.append({
            "id": row[0],
            "name": row[1],
            "document_count": row[2],
            "chunk_count": row[3]
        })
    
    conn.close()
    return projects


def get_vector_stats(project_id: str) -> Dict[str, int]:
    """获取项目向量统计"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 向量状态统计
    cursor.execute("""
        SELECT vector_status, COUNT(*) 
        FROM chunks 
        WHERE project_id = ?
        GROUP BY vector_status
    """, (project_id,))
    stats = dict(cursor.fetchall())
    
    # 有 vector_id 的数量
    cursor.execute("""
        SELECT COUNT(*) FROM chunks 
        WHERE project_id = ? AND vector_id IS NOT NULL
    """, (project_id,))
    stats["with_vector_id"] = cursor.fetchone()[0]
    
    conn.close()
    return stats


def get_qdrant_count(project_id: str) -> int:
    """获取 Qdrant 向量数量"""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host='localhost', port=6333, check_compatibility=False)
        
        collection_name = f"project_{project_id}"
        info = client.get_collection(collection_name)
        return info.points_count
    except:
        return 0


def check_and_fix_missing_vectors(project_id: str) -> Dict[str, int]:
    """检查并修复缺失的向量"""
    from scripts.sync_missing_vectors import sync_project, check_consistency
    
    result = check_consistency(project_id)
    
    if result["missing_in_qdrant"] > 0:
        logger.info(f"项目 {project_id}: 缺失 {result['missing_in_qdrant']} 个向量，开始同步...")
        sync_result = sync_project(project_id)
        return {
            "missing": result["missing_in_qdrant"],
            "synced": sync_result["synced"],
            "failed": sync_result.get("failed", 0)
        }
    
    return {"missing": 0, "synced": 0, "failed": 0}


def check_and_clean_orphans(project_id: str) -> int:
    """检查并清理孤儿向量"""
    from scripts.sync_missing_vectors import fix_orphan_vectors
    
    result = fix_orphan_vectors(project_id, dry_run=False)
    return result


def reset_failed_for_retry(project_id: str = None) -> int:
    """重置失败的 chunks 为 pending"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if project_id:
        cursor.execute("""
            UPDATE chunks 
            SET vector_status = 'pending', vector_error = NULL
            WHERE project_id = ? 
            AND vector_status = 'failed'
            AND vector_retry_count < ?
        """, (project_id, MAX_RETRY))
    else:
        cursor.execute("""
            UPDATE chunks 
            SET vector_status = 'pending', vector_error = NULL
            WHERE vector_status = 'failed'
            AND vector_retry_count < ?
        """, (MAX_RETRY,))
    
    count = cursor.rowcount
    conn.commit()
    conn.close()
    
    return count


def generate_report(results: Dict[str, Any]) -> str:
    """生成报告"""
    report_lines = [
        f"# RAG 向量同步报告",
        f"",
        f"**时间**: {datetime.now().isoformat()}",
        f"",
        f"## 项目统计",
        f"",
        f"| 项目名称 | SQLite向量 | Qdrant向量 | 缺失 | 孤儿 | 状态 |",
        f"|----------|------------|------------|------|------|------|",
    ]
    
    for project in results["projects"]:
        status = "✅" if project["missing"] == 0 and project["orphan"] == 0 else "⚠️"
        report_lines.append(
            f"| {project['name']} | {project['sqlite_count']} | "
            f"{project['qdrant_count']} | {project['missing']} | "
            f"{project['orphan']} | {status} |"
        )
    
    report_lines.extend([
        f"",
        f"## 执行结果",
        f"",
        f"- 检查项目数: {results['total_projects']}",
        f"- 缺失向量: {results['total_missing']}",
        f"- 已同步: {results['total_synced']}",
        f"- 孤儿向量: {results['total_orphans']}",
        f"- 已清理: {results['total_cleaned']}",
        f"- 失败重试: {results['total_reset']}",
        f"",
        f"## 下一步操作",
        f"",
    ])
    
    if results["total_missing"] > results["total_synced"]:
        report_lines.append(f"- [ ] 手动检查同步失败的向量")
    
    if results["total_orphans"] > results["total_cleaned"]:
        report_lines.append(f"- [ ] 手动清理残留孤儿向量")
    
    report_lines.append(f"- [ ] 检查失败向量详情: `python scripts/retry_failed_vectors.py --list`")
    
    return "\n".join(report_lines)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("RAG 向量同步任务开始")
    logger.info("=" * 60)
    
    # 创建报告目录
    REPORT_DIR.mkdir(exist_ok=True)
    
    # 获取所有项目
    projects = get_all_projects()
    logger.info(f"共 {len(projects)} 个项目")
    
    # 结果收集
    results = {
        "total_projects": len(projects),
        "total_missing": 0,
        "total_synced": 0,
        "total_orphans": 0,
        "total_cleaned": 0,
        "total_reset": 0,
        "projects": []
    }
    
    for project in projects:
        project_id = project["id"]
        project_name = project["name"]
        
        logger.info(f"\n--- 项目: {project_name} ---")
        
        # 获取统计
        stats = get_vector_stats(project_id)
        sqlite_count = stats.get("with_vector_id", 0)
        qdrant_count = get_qdrant_count(project_id)
        
        missing = max(0, sqlite_count - qdrant_count)
        orphan = max(0, qdrant_count - sqlite_count)
        
        project_result = {
            "id": project_id,
            "name": project_name,
            "sqlite_count": sqlite_count,
            "qdrant_count": qdrant_count,
            "missing": missing,
            "orphan": orphan,
            "synced": 0,
            "cleaned": 0
        }
        
        # 同步缺失向量
        if missing > 0:
            sync_result = check_and_fix_missing_vectors(project_id)
            project_result["synced"] = sync_result["synced"]
            results["total_missing"] += missing
            results["total_synced"] += sync_result["synced"]
        
        # 清理孤儿向量
        if orphan > 0:
            cleaned = check_and_clean_orphans(project_id)
            project_result["cleaned"] = cleaned
            results["total_orphans"] += orphan
            results["total_cleaned"] += cleaned
        
        results["projects"].append(project_result)
    
    # 重置失败的向量
    logger.info("\n--- 重置失败向量 ---")
    reset_count = reset_failed_for_retry()
    results["total_reset"] = reset_count
    logger.info(f"重置 {reset_count} 个失败向量为 pending")
    
    # 生成报告
    report = generate_report(results)
    report_path = REPORT_DIR / f"sync_report_{datetime.now().strftime('%Y%m%d')}.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"\n报告已保存: {report_path}")
    
    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("摘要:")
    logger.info(f"  检查项目: {results['total_projects']}")
    logger.info(f"  缺失向量: {results['total_missing']}")
    logger.info(f"  已同步: {results['total_synced']}")
    logger.info(f"  孤儿向量: {results['total_orphans']}")
    logger.info(f"  已清理: {results['total_cleaned']}")
    logger.info(f"  失败重试: {results['total_reset']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()