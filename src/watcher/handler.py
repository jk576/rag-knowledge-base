"""文件变更处理器 - 使用线程安全的防抖实现"""

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from watchdog.events import (
    DirCreatedEvent, DirDeletedEvent, DirMovedEvent,
    FileCreatedEvent, FileDeletedEvent, FileModifiedEvent,
    FileMovedEvent, FileSystemEvent, FileSystemEventHandler,
)
from watchdog.observers import Observer

from src.watcher.gitignore import gitignore_cache
from src.watcher.sync import ConsistencyChecker, FileSync, ProjectMapping, SyncStats
from src.rag_api.models.database import get_db_session

logger = logging.getLogger(__name__)


@dataclass
class FileEvent:
    """文件事件数据类"""
    event_type: str
    src_path: Path
    dest_path: Optional[Path] = None
    timestamp: float = field(default_factory=time.time)
    is_directory: bool = False


class EventDebouncer:
    """事件防抖器 - 使用 threading.Timer 实现，完全线程安全"""
    
    def __init__(self, debounce_interval: float = 1.0, callback: Optional[Callable] = None):
        self.debounce_interval = debounce_interval
        self._callback = callback
        self._pending_events: Dict[Path, FileEvent] = {}
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._timer_lock = threading.Lock()  # 专门用于timer操作的锁
    
    def set_callback(self, callback: Callable) -> None:
        self._callback = callback
    
    def add_event(self, event: FileEvent) -> None:
        """线程安全添加事件"""
        with self._lock:
            key = event.src_path
            
            # 合并事件
            if key in self._pending_events:
                existing = self._pending_events[key]
                if existing.event_type == "created" and event.event_type == "modified":
                    return
                if existing.event_type == "deleted" and event.event_type == "created":
                    event.event_type = "modified"
            
            self._pending_events[key] = event
        
        # 在锁外操作timer，避免死锁
        self._reset_timer()
        
        logger.debug(f"Event added: {event.event_type} for {key}, will process in {self.debounce_interval}s")
    
    def _reset_timer(self) -> None:
        """重置定时器 - 使用单独的锁保护"""
        with self._timer_lock:
            # 取消现有定时器
            if self._timer:
                self._timer.cancel()
                self._timer = None
            
            # 创建新定时器
            self._timer = threading.Timer(
                self.debounce_interval,
                self._process_events
            )
            self._timer.start()
    
    def _process_events(self) -> None:
        """在定时器线程中处理事件"""
        # 先获取事件，再释放锁
        with self._lock:
            if not self._pending_events:
                return
            events = list(self._pending_events.values())
            self._pending_events.clear()
        
        # 在timer锁中清理timer引用
        with self._timer_lock:
            self._timer = None
        
        if self._callback and events:
            logger.info(f"Processing {len(events)} debounced events")
            try:
                self._callback(events)
            except Exception as e:
                logger.error(f"Error processing events: {e}")
    
    def flush(self) -> None:
        """立即处理所有待处理事件"""
        # 先取消timer
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
        
        # 再获取事件
        with self._lock:
            if not self._pending_events:
                return
            events = list(self._pending_events.values())
            self._pending_events.clear()
        
        if self._callback and events:
            try:
                self._callback(events)
            except Exception as e:
                logger.error(f"Error flushing events: {e}")
    
    def clear(self) -> None:
        """清空所有待处理事件"""
        with self._timer_lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
        
        with self._lock:
            self._pending_events.clear()


class FileChangeHandler(FileSystemEventHandler):
    """文件变更事件处理器"""
    
    def __init__(
        self,
        watch_root: Path,
        project_name: str,
        db_session_factory: Callable,
        debounce_interval: float = 1.0,
    ):
        super().__init__()
        
        self.watch_root = Path(watch_root).resolve()
        self.project_name = project_name
        self.db_session_factory = db_session_factory
        self.gitignore = gitignore_cache.get_parser(self.watch_root)
        self.stats = SyncStats()
        
        # 初始化防抖器 - 使用 threading.Timer
        self.debouncer = EventDebouncer(debounce_interval)
        self.debouncer.set_callback(self._process_batch_events)
        
        self._processing = False
        
        logger.info(f"Initialized handler for project '{project_name}' watching {watch_root}")
    
    def _is_ignored(self, path: Path) -> bool:
        return self.gitignore.is_ignored(path)
    
    def _get_relative_path(self, path: Path) -> Optional[str]:
        try:
            return str(Path(path).relative_to(self.watch_root))
        except ValueError:
            return None
    
    def _handle_event(self, event: FileSystemEvent, event_type: str) -> None:
        """通用事件处理"""
        src_path = Path(event.src_path)
        
        if event_type != "deleted" and self._is_ignored(src_path):
            return
        
        if event.is_directory and event_type in ("modified",):
            return
        
        file_event = FileEvent(
            event_type=event_type,
            src_path=src_path,
            dest_path=Path(event.dest_path) if hasattr(event, 'dest_path') and event.dest_path else None,
            is_directory=event.is_directory,
        )
        self.debouncer.add_event(file_event)
        logger.debug(f"{event_type}: {event.src_path}")
    
    def on_created(self, event: FileSystemEvent) -> None:
        self._handle_event(event, "created")
    
    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle_event(event, "modified")
    
    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle_event(event, "deleted")
    
    def on_moved(self, event: FileSystemEvent) -> None:
        src_ignored = self._is_ignored(Path(event.src_path))
        dest_ignored = self._is_ignored(Path(event.dest_path))
        
        if src_ignored and not dest_ignored:
            self._handle_event(event, "created")
        elif not src_ignored and dest_ignored:
            self._handle_event(event, "deleted")
        elif not src_ignored and not dest_ignored:
            self._handle_event(event, "moved")
    
    def _process_batch_events(self, events: List[FileEvent]) -> None:
        """批量处理事件 - 在定时器线程中执行"""
        if self._processing:
            logger.warning("Previous batch still processing")
            return
        
        self._processing = True
        
        try:
            # 使用上下文管理器，自动处理commit/rollback/close
            with get_db_session() as db:
                project_mapping = ProjectMapping(db)
                project = project_mapping.get_or_create_project(self.watch_root, self.project_name)
                file_sync = FileSync(db, str(project.id))
                
                for event in events:
                    try:
                        self._process_single_event(event, file_sync)
                    except Exception as e:
                        logger.error(f"Error processing event {event}: {e}")
                        self.stats.errors += 1
                
                from datetime import datetime
                self.stats.last_sync = datetime.now()
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
        finally:
            self._processing = False
    
    def _process_single_event(self, event: FileEvent, file_sync: FileSync) -> None:
        """处理单个事件（同步）"""
        if event.is_directory:
            return
        
        rel_path = self._get_relative_path(event.src_path)
        if rel_path is None:
            return
        
        if not self.gitignore.should_process(event.src_path):
            self.stats.skipped += 1
            return
        
        try:
            if event.event_type in ("created", "modified"):
                # 使用同步方法
                result = file_sync.sync_file(event.src_path, rel_path)
                    
                if result["status"] == "created":
                    self.stats.created += 1
                elif result["status"] == "updated":
                    self.stats.updated += 1
                elif result["status"] == "skipped":
                    self.stats.skipped += 1
                elif result["status"] == "error":
                    self.stats.errors += 1
                    
            elif event.event_type == "deleted":
                result = file_sync.delete_file(rel_path)
                if result["status"] == "deleted":
                    self.stats.deleted += 1
                elif result["status"] == "skipped":
                    self.stats.skipped += 1
                elif result["status"] == "error":
                    self.stats.errors += 1
                    
            elif event.event_type == "moved":
                dest_rel_path = self._get_relative_path(event.dest_path) if event.dest_path else None
                if dest_rel_path and self.gitignore.should_process(event.dest_path):
                    result = file_sync.rename_file(rel_path, dest_rel_path)
                    if result["status"] == "renamed":
                        self.stats.renamed += 1
        except Exception as e:
            logger.error(f"Error processing event {event}: {e}")
            self.stats.errors += 1
    
    def _scan_directory(self, directory: Path) -> None:
        """扫描目录（同步）"""
        if not directory.exists():
            return
        
        try:
            # 使用上下文管理器，自动处理commit/rollback/close
            with get_db_session() as db:
                project_mapping = ProjectMapping(db)
                project = project_mapping.get_or_create_project(self.watch_root, self.project_name)
                
                # 首先执行一致性检查（清理孤儿文件）
                checker = ConsistencyChecker(db, str(project.id), self.watch_root)
                check_stats = checker.check_and_fix()
                
                if check_stats['orphaned_files'] > 0:
                    logger.info(f"一致性检查: 清理了 {check_stats['cleaned']}/{check_stats['orphaned_files']} 个孤儿文件")
                
                # 然后执行正常同步
                file_sync = FileSync(db, str(project.id))
                
                for file_path in directory.rglob("*"):
                    if file_path.is_file() and self.gitignore.should_process(file_path):
                        rel_path = self._get_relative_path(file_path)
                        if rel_path:
                            try:
                                # 使用同步方法
                                result = file_sync.sync_file(file_path, rel_path)
                                if result["status"] == "created":
                                    self.stats.created += 1
                                elif result["status"] == "updated":
                                    self.stats.updated += 1
                            except Exception as e:
                                logger.error(f"Error syncing file {file_path}: {e}")
                                self.stats.errors += 1
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "watch_root": str(self.watch_root),
            **self.stats.to_dict(),
        }
    
    def flush(self) -> None:
        self.debouncer.flush()
    
    def reset_stats(self) -> None:
        self.stats.reset()


class ProjectDirectoryHandler:
    """项目目录处理器"""
    
    def __init__(self, projects_root: Path, db_session_factory: Callable):
        self.projects_root = Path(projects_root).resolve()
        self.db_session_factory = db_session_factory
        self.project_handlers: Dict[str, FileChangeHandler] = {}
        self.observer: Optional[Observer] = None
    
    def _is_valid_project_dir(self, path: Path) -> bool:
        try:
            rel_path = path.relative_to(self.projects_root)
            parts = rel_path.parts
            if len(parts) != 1 or not path.is_dir() or parts[0].startswith("."):
                return False
            return True
        except ValueError:
            return False
    
    def on_project_created(self, project_path: Path) -> None:
        if not self._is_valid_project_dir(project_path):
            return
        project_name = project_path.name
        logger.info(f"New project: {project_name}")
        
        handler = FileChangeHandler(
            watch_root=project_path,
            project_name=project_name,
            db_session_factory=self.db_session_factory,
        )
        self.project_handlers[project_name] = handler
        
        if self.observer:
            self.observer.schedule(handler, str(project_path), recursive=True)
        
        handler._scan_directory(project_path)
    
    def on_project_deleted(self, project_path: Path) -> None:
        if not self._is_valid_project_dir(project_path):
            return
        project_name = project_path.name
        
        if project_name in self.project_handlers:
            del self.project_handlers[project_name]
        
        try:
            with get_db_session() as db:
                project_mapping = ProjectMapping(db)
                project_mapping.delete_project_by_name(project_name)
        except Exception as e:
            logger.error(f"Error deleting project {project_name}: {e}")
    
    def on_project_moved(self, src_path: Path, dest_path: Path) -> None:
        old_name = src_path.name
        new_name = dest_path.name
        
        if not self._is_valid_project_dir(src_path) or not self._is_valid_project_dir(dest_path):
            return
        
        if old_name in self.project_handlers:
            handler = self.project_handlers.pop(old_name)
            handler.project_name = new_name
            handler.watch_root = dest_path
            self.project_handlers[new_name] = handler
        
        try:
            with get_db_session() as db:
                project_mapping = ProjectMapping(db)
                project_mapping.update_project_name(old_name, new_name)
        except Exception as e:
            logger.error(f"Error moving project {old_name} to {new_name}: {e}")
    
    def get_all_project_paths(self) -> List[Path]:
        if not self.projects_root.exists():
            return []
        return [item for item in self.projects_root.iterdir() if self._is_valid_project_dir(item)]
