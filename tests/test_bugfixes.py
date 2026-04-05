"""
RAG 系统 Bug 修复验证测试

测试范围:
1. 数据库连接管理
2. EventDebouncer 线程安全
3. EmbeddingService 内存管理

运行方式:
    cd ~/Projects/rag-knowledge-base
    python -m pytest tests/test_bugfixes.py -v
"""

import sys
import os
import threading
import time
import gc
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from sqlalchemy import text


# =============================================================================
# 测试 1: 数据库连接管理
# =============================================================================

class TestDatabaseConnection:
    """测试数据库连接管理修复"""
    
    def test_get_db_session_context_manager(self):
        """测试上下文管理器正常工作"""
        from src.rag_api.models.database import get_db_session
        
        with get_db_session() as db:
            result = db.execute(text("SELECT 1")).fetchone()
            assert result[0] == 1
        # 自动关闭，不抛异常
    
    def test_get_db_session_auto_rollback(self):
        """测试异常时自动回滚"""
        from src.rag_api.models.database import get_db_session
        from src.rag_api.models.database import Project
        
        initial_count = 0
        with get_db_session() as db:
            initial_count = db.query(Project).count()
        
        try:
            with get_db_session() as db:
                # 创建临时项目
                project = Project(name="test_rollback_project")
                db.add(project)
                db.flush()
                # 抛出异常，应自动回滚
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # 验证项目未创建
        with get_db_session() as db:
            final_count = db.query(Project).count()
            assert final_count == initial_count, "事务未正确回滚"
    
    def test_concurrent_db_access(self):
        """测试并发数据库访问"""
        from src.rag_api.models.database import get_db_session
        
        results = []
        errors = []
        
        def query_db(thread_id):
            try:
                with get_db_session() as db:
                    result = db.execute(text("SELECT 1 as val")).fetchone()
                    results.append((thread_id, result[0]))
                    time.sleep(0.01)  # 模拟一些工作
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # 并发执行
        threads = []
        for i in range(20):
            t = threading.Thread(target=query_db, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"并发访问出错: {errors}"
        assert len(results) == 20, f"预期20个结果，实际{len(results)}"
    
    def test_connection_pool_limits(self):
        """测试连接池限制"""
        from src.rag_api.models.database import engine
        
        # 检查连接池配置
        assert engine.pool.size() >= 5, "连接池大小不足"
        print(f"✅ 连接池配置: size={engine.pool.size()}")


# =============================================================================
# 测试 2: EventDebouncer 线程安全
# =============================================================================

class TestEventDebouncer:
    """测试 EventDebouncer 线程安全"""
    
    def test_concurrent_event_addition(self):
        """测试并发添加事件"""
        from src.watcher.handler import EventDebouncer, FileEvent
        from pathlib import Path
        
        processed_events = []
        
        def callback(events):
            processed_events.extend(events)
        
        debouncer = EventDebouncer(debounce_interval=0.1)
        debouncer.set_callback(callback)
        
        # 并发添加事件
        def add_events(thread_id):
            for i in range(10):
                event = FileEvent(
                    event_type="modified",
                    src_path=Path(f"/tmp/test_{thread_id}_{i}.txt")
                )
                debouncer.add_event(event)
                time.sleep(0.001)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=add_events, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # 等待防抖处理
        time.sleep(0.5)
        debouncer.flush()
        
        # 验证无异常，事件被处理
        print(f"✅ 并发添加事件: {len(processed_events)} 个事件被处理")
    
    def test_timer_thread_safety(self):
        """测试定时器线程安全"""
        from src.watcher.handler import EventDebouncer, FileEvent
        from pathlib import Path
        
        callback_count = [0]
        
        def callback(events):
            callback_count[0] += 1
        
        debouncer = EventDebouncer(debounce_interval=0.05)
        debouncer.set_callback(callback)
        
        # 快速连续添加事件，触发定时器重置
        for i in range(20):
            event = FileEvent(
                event_type="modified",
                src_path=Path(f"/tmp/test_{i}.txt")
            )
            debouncer.add_event(event)
            time.sleep(0.02)  # 小于防抖间隔，会不断重置
        
        # 等待最终处理
        time.sleep(0.2)
        debouncer.flush()
        
        # 验证回调被正确执行（可能被合并为少数几次）
        assert callback_count[0] >= 1, "回调未被触发"
        print(f"✅ 定时器线程安全: {callback_count[0]} 次回调")
    
    def test_flush_during_processing(self):
        """测试处理期间flush"""
        from src.watcher.handler import EventDebouncer, FileEvent
        from pathlib import Path
        
        processing_started = threading.Event()
        can_finish = threading.Event()
        
        def slow_callback(events):
            processing_started.set()
            can_finish.wait(timeout=1.0)  # 模拟慢处理
        
        debouncer = EventDebouncer(debounce_interval=0.01)
        debouncer.set_callback(slow_callback)
        
        # 添加事件触发处理
        debouncer.add_event(FileEvent(
            event_type="modified",
            src_path=Path("/tmp/test1.txt")
        ))
        
        # 等待处理开始
        processing_started.wait(timeout=0.5)
        
        # 处理期间添加更多事件并flush
        debouncer.add_event(FileEvent(
            event_type="modified",
            src_path=Path("/tmp/test2.txt")
        ))
        
        # 不应死锁
        debouncer.flush()
        
        can_finish.set()
        print("✅ 处理期间flush无死锁")


# =============================================================================
# 测试 3: EmbeddingService 内存管理
# =============================================================================

class TestEmbeddingService:
    """测试 EmbeddingService 内存管理"""
    
    def test_no_memory_leak_on_repeated_creation(self):
        """测试重复创建不泄漏内存"""
        from src.core.embedding import EmbeddingService
        
        process = psutil.Process()
        
        # 获取初始内存
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 重复创建和关闭服务
        for i in range(10):
            service = EmbeddingService()
            # 不调用close，模拟之前的__del__行为
            del service
            gc.collect()
        
        # 检查内存增长
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        print(f"✅ 内存增长: {memory_growth:.2f} MB")
        # 允许一定增长，但不应持续累积
        assert memory_growth < 50, f"内存泄漏: 增长{memory_growth:.2f} MB"
    
    def test_explicit_close(self):
        """测试显式关闭"""
        from src.core.embedding import EmbeddingService
        
        service = EmbeddingService()
        
        # 创建客户端
        _ = service.async_client
        _ = service.sync_client
        
        # 显式关闭
        import asyncio
        asyncio.run(service.close())
        
        # 验证客户端已关闭
        assert service._async_client is None
        assert service._sync_client is None
        print("✅ 显式关闭成功")
    
    def test_no_del_method(self):
        """测试__del__方法已移除"""
        from src.core.embedding import EmbeddingService
        
        # 验证__del__不存在
        assert not hasattr(EmbeddingService, "__del__") or \
               EmbeddingService.__del__ is object.__del__, \
            "__del__方法应已移除"
        print("✅ __del__方法已移除")


# =============================================================================
# 测试 4: 集成测试
# =============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_watcher_handler_with_new_db_session(self):
        """测试 Watcher Handler 使用新的数据库会话管理"""
        from src.watcher.handler import FileChangeHandler
        from src.rag_api.models.database import get_db_session
        from pathlib import Path
        import tempfile
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            handler = FileChangeHandler(
                watch_root=Path(tmpdir),
                project_name="test_project",
                db_session_factory=get_db_session,  # 传递新的上下文管理器
                debounce_interval=0.1
            )
            
            # 验证初始化成功
            assert handler.project_name == "test_project"
            print("✅ Watcher Handler 初始化成功")


# =============================================================================
# 主函数
# =============================================================================

def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("RAG 系统 Bug 修复验证测试")
    print("=" * 60)
    
    test_classes = [
        TestDatabaseConnection,
        TestEventDebouncer,
        TestEmbeddingService,
        TestIntegration,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"测试类: {test_class.__name__}")
        print('=' * 60)
        
        instance = test_class()
        methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                print(f"\n  运行: {method_name}...")
                method()
                passed_tests += 1
                print(f"  ✅ 通过")
            except Exception as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"  ❌ 失败: {e}")
    
    print(f"\n{'='*60}")
    print("测试结果汇总")
    print('=' * 60)
    print(f"总测试数: {total_tests}")
    print(f"通过: {passed_tests}")
    print(f"失败: {len(failed_tests)}")
    
    if failed_tests:
        print("\n失败的测试:")
        for cls, method, error in failed_tests:
            print(f"  - {cls}.{method}: {error}")
        return False
    else:
        print("\n🎉 所有测试通过！")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)