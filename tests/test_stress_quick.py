"""
RAG 系统快速压力测试 (30秒内完成)
"""

import sys
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_api.models.database import get_db_session
from src.watcher.handler import EventDebouncer, FileEvent


def test_db_concurrent():
    """快速数据库并发测试"""
    print("\n测试 1: 数据库并发 (50线程 x 5次)")
    
    results = {"success": 0, "error": 0}
    
    def db_op(tid):
        try:
            with get_db_session() as db:
                from src.rag_api.models.database import Project
                _ = db.query(Project).count()
            results["success"] += 1
            return True
        except Exception as e:
            results["error"] += 1
            return False
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        list(executor.map(db_op, range(50)))
    
    print(f"  成功: {results['success']}/50, 错误: {results['error']}")
    assert results['error'] == 0
    print("  ✅ 通过")


def test_event_debouncer():
    """快速EventDebouncer测试"""
    print("\n测试 2: EventDebouncer (20线程 x 20事件)")
    
    processed = [0]
    
    def callback(events):
        processed[0] += len(events)
    
    debouncer = EventDebouncer(debounce_interval=0.05)
    debouncer.set_callback(callback)
    
    def add_batch(bid):
        for i in range(20):
            debouncer.add_event(FileEvent(
                event_type="modified",
                src_path=Path(f"/tmp/t{bid}_{i}.txt")
            ))
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        list(executor.map(add_batch, range(20)))
    
    time.sleep(0.3)
    debouncer.flush()
    time.sleep(0.1)
    
    print(f"  发送: 400, 处理: {processed[0]}")
    assert processed[0] > 0
    print("  ✅ 通过")


def test_mixed():
    """快速混合测试"""
    print("\n测试 3: 混合操作 (50线程)")
    
    results = {"ok": 0, "err": 0}
    debouncer = EventDebouncer(debounce_interval=0.1)
    debouncer.set_callback(lambda e: None)
    
    def mixed_op(oid):
        try:
            if oid % 2 == 0:
                with get_db_session() as db:
                    from src.rag_api.models.database import Project
                    _ = db.query(Project).first()
            else:
                debouncer.add_event(FileEvent(
                    event_type="modified",
                    src_path=Path(f"/tmp/m{oid}.txt")
                ))
            results["ok"] += 1
        except:
            results["err"] += 1
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        list(executor.map(mixed_op, range(50)))
    
    print(f"  成功: {results['ok']}/50, 错误: {results['err']}")
    assert results['err'] == 0
    print("  ✅ 通过")


def main():
    print("="*50)
    print("RAG 系统快速压力测试")
    print("="*50)
    
    start = time.time()
    
    test_db_concurrent()
    test_event_debouncer()
    test_mixed()
    
    elapsed = time.time() - start
    
    print(f"\n{'='*50}")
    print(f"🎉 全部通过! 耗时: {elapsed:.1f}s")
    print("="*50)


if __name__ == "__main__":
    main()