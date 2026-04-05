"""
RAG 系统压力测试

测试高并发场景下的稳定性

运行方式:
    cd ~/Projects/rag-knowledge-base
    python tests/test_stress.py
"""

import sys
import threading
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag_api.models.database import get_db_session
from src.watcher.handler import EventDebouncer, FileEvent


def stress_test_database_connections():
    """压力测试：数据库连接"""
    print("\n" + "="*60)
    print("压力测试 1: 数据库连接并发")
    print("="*60)
    
    results = {"success": 0, "error": 0}
    latencies = []
    
    def db_operation(thread_id):
        try:
            start = time.time()
            with get_db_session() as db:
                # 模拟一些数据库操作
                from src.rag_api.models.database import Project
                count = db.query(Project).count()
                time.sleep(0.001)  # 模拟处理时间
            
            latency = time.time() - start
            latencies.append(latency)
            results["success"] += 1
            return True
        except Exception as e:
            results["error"] += 1
            print(f"  线程 {thread_id} 错误: {e}")
            return False
    
    # 并发100个线程，每个执行10次
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for i in range(100):
            for j in range(10):
                futures.append(executor.submit(db_operation, f"{i}-{j}"))
        
        for future in as_completed(futures):
            future.result()
    
    total_time = time.time() - start_time
    
    print(f"  总操作数: 1000")
    print(f"  成功: {results['success']}")
    print(f"  失败: {results['error']}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  平均延迟: {sum(latencies)/len(latencies)*1000:.2f}ms")
    print(f"  最大延迟: {max(latencies)*1000:.2f}ms")
    print(f"  吞吐量: {1000/total_time:.1f} ops/sec")
    
    assert results['error'] == 0, "有数据库操作失败"
    print("  ✅ 数据库连接压力测试通过")


def stress_test_event_debouncer():
    """压力测试：EventDebouncer"""
    print("\n" + "="*60)
    print("压力测试 2: EventDebouncer 高并发事件")
    print("="*60)
    
    processed_count = [0]
    error_count = [0]
    
    def callback(events):
        try:
            processed_count[0] += len(events)
            time.sleep(0.001)  # 模拟处理
        except Exception as e:
            error_count[0] += 1
    
    debouncer = EventDebouncer(debounce_interval=0.05)
    debouncer.set_callback(callback)
    
    def add_events_batch(batch_id):
        try:
            for i in range(50):
                event = FileEvent(
                    event_type=random.choice(["created", "modified", "deleted"]),
                    src_path=Path(f"/tmp/stress_test/{batch_id}_{i}.txt")
                )
                debouncer.add_event(event)
                time.sleep(random.uniform(0, 0.001))
        except Exception as e:
            error_count[0] += 1
            print(f"  批次 {batch_id} 错误: {e}")
    
    start_time = time.time()
    
    # 50个线程，每个添加50个事件
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(add_events_batch, i) for i in range(50)]
        for future in as_completed(futures):
            future.result()
    
    # 等待处理完成
    time.sleep(0.5)
    debouncer.flush()
    time.sleep(0.2)
    
    total_time = time.time() - start_time
    
    print(f"  总事件数: 2500")
    print(f"  处理事件数: {processed_count[0]}")
    print(f"  错误数: {error_count[0]}")
    print(f"  总耗时: {total_time:.2f}s")
    
    assert error_count[0] == 0, "有错误发生"
    assert processed_count[0] > 0, "事件未被处理"
    print("  ✅ EventDebouncer 压力测试通过")


def stress_test_mixed_operations():
    """压力测试：混合操作"""
    print("\n" + "="*60)
    print("压力测试 3: 混合操作")
    print("="*60)
    
    results = {"db_success": 0, "db_error": 0, "event_success": 0, "event_error": 0}
    
    # 创建共享的debouncer
    def event_callback(events):
        results["event_success"] += len(events)
    
    debouncer = EventDebouncer(debounce_interval=0.1)
    debouncer.set_callback(event_callback)
    
    def mixed_operation(op_id):
        try:
            # 50%概率执行数据库操作，50%概率添加事件
            if random.random() < 0.5:
                with get_db_session() as db:
                    from src.rag_api.models.database import Project
                    _ = db.query(Project).first()
                results["db_success"] += 1
            else:
                event = FileEvent(
                    event_type="modified",
                    src_path=Path(f"/tmp/mixed_test/{op_id}.txt")
                )
                debouncer.add_event(event)
                results["event_success"] += 1
        except Exception as e:
            if random.random() < 0.5:
                results["db_error"] += 1
            else:
                results["event_error"] += 1
            print(f"  操作 {op_id} 错误: {e}")
    
    start_time = time.time()
    
    # 200个并发操作
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = [executor.submit(mixed_operation, i) for i in range(200)]
        for future in as_completed(futures):
            future.result()
    
    # 等待事件处理
    time.sleep(0.5)
    debouncer.flush()
    
    total_time = time.time() - start_time
    
    total_ops = results["db_success"] + results["event_success"]
    total_errors = results["db_error"] + results["event_error"]
    
    print(f"  总操作数: 200")
    print(f"  成功: {total_ops}")
    print(f"  错误: {total_errors}")
    print(f"  总耗时: {total_time:.2f}s")
    print(f"  吞吐量: {200/total_time:.1f} ops/sec")
    
    assert total_errors == 0, "有操作失败"
    print("  ✅ 混合操作压力测试通过")


def stress_test_long_running():
    """压力测试：长时间运行"""
    print("\n" + "="*60)
    print("压力测试 4: 长时间运行 (10秒)")
    print("="*60)
    
    results = {"operations": 0, "errors": 0}
    stop_flag = threading.Event()
    
    def continuous_db_access():
        while not stop_flag.is_set():
            try:
                with get_db_session() as db:
                    from src.rag_api.models.database import Project
                    _ = db.query(Project).count()
                results["operations"] += 1
            except Exception as e:
                results["errors"] += 1
            time.sleep(0.01)
    
    # 启动10个线程持续运行
    threads = []
    for i in range(10):
        t = threading.Thread(target=continuous_db_access)
        t.start()
        threads.append(t)
    
    # 运行10秒
    time.sleep(10)
    stop_flag.set()
    
    for t in threads:
        t.join(timeout=2)
    
    print(f"  总操作数: {results['operations']}")
    print(f"  错误数: {results['errors']}")
    print(f"  平均速率: {results['operations']/10:.1f} ops/sec")
    
    assert results["errors"] == 0, "长时间运行出现错误"
    print("  ✅ 长时间运行压力测试通过")


def main():
    """运行所有压力测试"""
    print("="*60)
    print("RAG 系统压力测试")
    print("="*60)
    
    try:
        stress_test_database_connections()
        stress_test_event_debouncer()
        stress_test_mixed_operations()
        stress_test_long_running()
        
        print("\n" + "="*60)
        print("🎉 所有压力测试通过！")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n❌ 压力测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)