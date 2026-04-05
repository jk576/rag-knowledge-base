"""监控管理器

管理文件系统监控的启动、停止和状态查询。
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from watchdog.observers import Observer

from src.rag_api.models.database import SessionLocal, get_db_session
from src.watcher.handler import FileChangeHandler, ProjectDirectoryHandler
from src.watcher.sync import SyncStats

logger = logging.getLogger(__name__)

# 定期一致性检查间隔（秒）
CONSISTENCY_CHECK_INTERVAL = 3600  # 1小时
# 健康检查间隔（秒）
HEALTH_CHECK_INTERVAL = 300  # 5分钟


@dataclass
class WatcherStatus:
    """监控器状态"""
    is_running: bool = False
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    projects_root: str = ""
    watched_projects: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_running": self.is_running,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "projects_root": self.projects_root,
            "watched_projects": self.watched_projects,
            "error_count": len(self.errors),
            "recent_errors": self.errors[-5:] if self.errors else [],
        }


class WatcherManager:
    """监控管理器
    
    单例模式管理文件系统监控的生命周期。
    """
    
    _instance: Optional["WatcherManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, projects_root: Optional[Path] = None, debounce_interval: float = 1.0):
        """
        初始化监控管理器
        
        Args:
            projects_root: 项目根目录，默认 ~/Projects
            debounce_interval: 事件防抖间隔（秒）
        """
        if self._initialized:
            return
        
        self.projects_root = Path(projects_root or Path.home() / "Projects").resolve()
        self.debounce_interval = debounce_interval
        
        self._observer: Optional[Observer] = None
        self._status = WatcherStatus(projects_root=str(self.projects_root))
        self._project_handler: Optional[ProjectDirectoryHandler] = None
        self._project_handlers: Dict[str, FileChangeHandler] = {}
        self._stats: Dict[str, SyncStats] = {}
        
        # 待添加项目队列（Watcher停止时添加的项目）
        self._pending_add_projects: Set[str] = set()
        
        # 定期检查线程（在 start() 时初始化）
        self._consistency_check_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        
        self._initialized = True
        
        self._initialized = True
        logger.info(f"WatcherManager initialized with root: {self.projects_root}")
    
    def _scan_project(self, handler, project_path: Path, project_name: str) -> None:
        """在后台线程扫描项目目录"""
        try:
            handler._scan_directory(project_path)
            logger.info(f"Completed initial scan for project: {project_name}")
        except Exception as e:
            logger.error(f"Error scanning project {project_name}: {e}")
            self._status.errors.append(f"Scan error for {project_name}: {str(e)}")
    
    def _get_project_watcher_enabled(self, project_name: str) -> bool:
        """
        从数据库查询项目的 watcher_enabled 状态
        
        Args:
            project_name: 项目名称
            
        Returns:
            是否启用监控（如果项目不存在或出错，默认返回 False）
        """
        try:
            with get_db_session() as db:
                from src.rag_api.models.database import Project as ProjectModel
                project = db.query(ProjectModel).filter(
                    ProjectModel.name == project_name
                ).first()
                if project:
                    return bool(project.watcher_enabled)
                return False
        except Exception as e:
            logger.error(f"Error querying watcher_enabled for {project_name}: {e}")
            return False
    
    def _is_project_watched(self, project_name: str) -> bool:
        """检查项目是否正在被监控"""
        return project_name in self._project_handlers
    
    def add_project_watch(self, project_name: str) -> Dict[str, Any]:
        """
        添加项目到监控列表
        
        Args:
            project_name: 项目名称
            
        Returns:
            操作结果
        """
        if not self._status.is_running:
            return {
                "success": False,
                "message": "Watcher is not running",
            }
        
        if project_name in self._project_handlers:
            return {
                "success": False,
                "message": f"Project {project_name} is already being watched",
            }
        
        project_path = self.projects_root / project_name
        if not project_path.exists() or not project_path.is_dir():
            return {
                "success": False,
                "message": f"Project directory not found: {project_path}",
            }
        
        try:
            handler = FileChangeHandler(
                watch_root=project_path,
                project_name=project_name,
                db_session_factory=self._get_db_session,
                debounce_interval=self.debounce_interval,
            )
            
            self._project_handlers[project_name] = handler
            self._observer.schedule(handler, str(project_path), recursive=True)
            
            # 在后台线程执行初始扫描
            scan_thread = threading.Thread(
                target=self._scan_project,
                args=(handler, project_path, project_name),
                daemon=True
            )
            scan_thread.start()
            
            # 更新状态
            self._status.watched_projects = list(self._project_handlers.keys())
            
            logger.info(f"Added project to watch: {project_name}")
            return {
                "success": True,
                "message": f"Project {project_name} added to watch",
            }
        except Exception as e:
            logger.error(f"Error adding project {project_name} to watch: {e}")
            return {
                "success": False,
                "message": f"Failed to add project {project_name}: {str(e)}",
            }
    
    def remove_project_watch(self, project_name: str) -> Dict[str, Any]:
        """
        从监控列表移除项目
        
        注意：watchdog 不支持直接取消监控特定路径，所以我们只是停止处理该项目的
        事件。完全停止监控需要重启 observer。
        
        Args:
            project_name: 项目名称
            
        Returns:
            操作结果
        """
        if not self._status.is_running:
            return {
                "success": False,
                "message": "Watcher is not running",
            }
        
        if project_name not in self._project_handlers:
            return {
                "success": False,
                "message": f"Project {project_name} is not being watched",
            }
        
        try:
            # 从监控列表中移除
            handler = self._project_handlers.pop(project_name)
            
            # 刷新待处理的事件
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                future = asyncio.run_coroutine_threadsafe(handler.flush(), loop)
                future.result(timeout=5)
            except Exception as e:
                logger.warning(f"Error flushing handler for {project_name}: {e}")
            
            # 更新状态
            self._status.watched_projects = list(self._project_handlers.keys())
            
            logger.info(f"Removed project from watch: {project_name}")
            return {
                "success": True,
                "message": f"Project {project_name} removed from watch",
            }
        except Exception as e:
            logger.error(f"Error removing project {project_name} from watch: {e}")
            return {
                "success": False,
                "message": f"Failed to remove project {project_name}: {str(e)}",
            }
    
    def refresh_project_watch(self, project_name: str, watcher_enabled: bool) -> Dict[str, Any]:
        """
        根据 watcher_enabled 状态刷新项目的监控状态
        
        如果 Watcher 未运行，将项目加入待添加队列，启动时自动处理。
        
        Args:
            project_name: 项目名称
            watcher_enabled: 是否启用监控
            
        Returns:
            操作结果
        """
        is_watched = self._is_project_watched(project_name)
        
        # Watcher 未运行时，处理待添加队列
        if not self._status.is_running:
            if watcher_enabled:
                self._pending_add_projects.add(project_name)
                return {
                    "success": True,
                    "message": f"Project {project_name} queued for watch (watcher not running)",
                }
            else:
                # 从待添加队列中移除
                self._pending_add_projects.discard(project_name)
                return {
                    "success": True,
                    "message": f"Project {project_name} removed from pending queue (watcher not running)",
                }
        
        # Watcher 运行时，直接处理
        if watcher_enabled and not is_watched:
            return self.add_project_watch(project_name)
        elif not watcher_enabled and is_watched:
            return self.remove_project_watch(project_name)
        else:
            return {
                "success": True,
                "message": f"Project {project_name} watch status unchanged (watched={is_watched}, enabled={watcher_enabled})",
            }
    def start(self) -> Dict[str, Any]:
        """
        启动文件系统监控
        
        只监控数据库中 watcher_enabled=1 的项目。
        
        Returns:
            操作结果
        """
        if self._status.is_running:
            return {
                "success": False,
                "message": "Watcher is already running",
                "status": self._status.to_dict(),
            }
        
        try:
            # 检查目录是否存在
            if not self.projects_root.exists():
                logger.warning(f"Projects root does not exist: {self.projects_root}")
                self.projects_root.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created projects root: {self.projects_root}")
            
            # 创建项目目录处理器
            self._project_handler = ProjectDirectoryHandler(
                projects_root=self.projects_root,
                db_session_factory=self._get_db_session,
            )
            
            # 创建并启动 observer
            self._observer = Observer()
            
            # 首先扫描现有的项目目录
            existing_projects = self._project_handler.get_all_project_paths()
            logger.info(f"Found {len(existing_projects)} existing project directories")
            
            # 只为 watcher_enabled=1 的项目创建处理器并开始监控
            scan_threads = []
            watched_count = 0
            skipped_count = 0
            
            for project_path in existing_projects:
                project_name = project_path.name
                
                # 检查项目是否存在于数据库，如果不存在则自动创建
                from src.watcher.sync import ProjectMapping
                try:
                    with get_db_session() as db:
                        project_mapping = ProjectMapping(db)
                        project = project_mapping.get_project_by_name(project_name)
                        if not project:
                            # 新项目：自动创建并启用监控
                            project = project_mapping.get_or_create_project(project_path, project_name)
                            logger.info(f"Auto-created new project: {project_name} (watcher_enabled={project.watcher_enabled})")
                except Exception as e:
                    logger.error(f"Error checking/creating project {project_name}: {e}")
                    continue
                
                # 检查项目的 watcher_enabled 状态
                if not self._get_project_watcher_enabled(project_name):
                    logger.info(f"Skipping project {project_name}: watcher_enabled=false")
                    skipped_count += 1
                    continue
                
                handler = FileChangeHandler(
                    watch_root=project_path,
                    project_name=project_name,
                    db_session_factory=get_db_session,  # 传递上下文管理器
                    debounce_interval=self.debounce_interval,
                )
                
                self._project_handlers[project_name] = handler
                self._observer.schedule(handler, str(project_path), recursive=True)
                
                # 在后台线程执行初始扫描，避免阻塞 API 响应
                scan_thread = threading.Thread(
                    target=self._scan_project,
                    args=(handler, project_path, project_name),
                    daemon=True
                )
                scan_threads.append(scan_thread)
                scan_thread.start()
                
                watched_count += 1
                logger.info(f"Started watching project: {project_name}")
            
            # 监控 Projects 根目录（用于检测新项目创建/删除/重命名）
            root_handler = self._create_root_handler()
            self._observer.schedule(root_handler, str(self.projects_root), recursive=False)
            
            # 启动 observer
            self._observer.start()
            
            # 处理待添加队列（Watcher停止时添加的项目）
            if self._pending_add_projects:
                pending_added = 0
                for project_name in list(self._pending_add_projects):
                    result = self.add_project_watch(project_name)
                    if result["success"]:
                        pending_added += 1
                        logger.info(f"Added pending project: {project_name}")
                    else:
                        logger.warning(f"Failed to add pending project {project_name}: {result['message']}")
                    self._pending_add_projects.discard(project_name)
                if pending_added > 0:
                    logger.info(f"Added {pending_added} projects from pending queue")
            
            # 更新状态
            self._status.is_running = True
            self._status.started_at = datetime.utcnow()
            self._status.stopped_at = None
            self._status.projects_root = str(self.projects_root)
            self._status.watched_projects = list(self._project_handlers.keys())
            self._status.errors = []
            
            # 启动定期检查线程
            self.start_periodic_checks()
            
            logger.info(f"Watcher started successfully, monitoring {watched_count} projects (skipped {skipped_count})")
            
            return {
                "success": True,
                "message": f"Watcher started, monitoring {watched_count} projects (skipped {skipped_count})",
                "status": self._status.to_dict(),
            }
            
        except Exception as e:
            error_msg = f"Failed to start watcher: {str(e)}"
            logger.error(error_msg)
            self._status.errors.append(error_msg)
            
            # 清理
            self._stop_internal()
            
            return {
                "success": False,
                "message": error_msg,
                "status": self._status.to_dict(),
            }
    
    def stop(self) -> Dict[str, Any]:
        """
        停止文件系统监控
        
        Returns:
            操作结果
        """
        if not self._status.is_running:
            return {
                "success": False,
                "message": "Watcher is not running",
                "status": self._status.to_dict(),
            }
        
        try:
            # 先停止定期检查线程
            self.stop_periodic_checks()
            
            self._stop_internal()
            
            self._status.is_running = False
            self._status.stopped_at = datetime.utcnow()
            self._status.watched_projects = []
            
            logger.info("Watcher stopped successfully")
            
            return {
                "success": True,
                "message": "Watcher stopped",
                "status": self._status.to_dict(),
            }
            
        except Exception as e:
            error_msg = f"Error stopping watcher: {str(e)}"
            logger.error(error_msg)
            self._status.errors.append(error_msg)
            
            return {
                "success": False,
                "message": error_msg,
                "status": self._status.to_dict(),
            }
    
    def _stop_internal(self) -> None:
        """内部停止方法"""
        # 停止 observer
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping observer: {e}")
            finally:
                self._observer = None
        
        # 清理处理器
        self._project_handlers.clear()
        self._project_handler = None
    
    def _create_root_handler(self):
        """创建根目录处理器"""
        from watchdog.events import FileSystemEventHandler, DirCreatedEvent, DirDeletedEvent, DirMovedEvent
        
        class RootHandler(FileSystemEventHandler):
            def __init__(self, manager: "WatcherManager"):
                self.manager = manager
            
            def on_created(self, event):
                if isinstance(event, DirCreatedEvent):
                    self.manager._handle_project_created(Path(event.src_path))
            
            def on_deleted(self, event):
                if isinstance(event, DirDeletedEvent):
                    self.manager._handle_project_deleted(Path(event.src_path))
            
            def on_moved(self, event):
                if isinstance(event, DirMovedEvent):
                    self.manager._handle_project_moved(Path(event.src_path), Path(event.dest_path))
        
        return RootHandler(self)
    
    def _handle_project_created(self, project_path: Path) -> None:
        """处理新项目创建"""
        if not self._project_handler:
            return
        
        if not self._project_handler._is_valid_project_dir(project_path):
            return
        
        project_name = project_path.name
        
        # 检查项目是否存在于数据库，如果不存在则自动创建
        from src.watcher.sync import ProjectMapping
        try:
            with get_db_session() as db:
                project_mapping = ProjectMapping(db)
                project = project_mapping.get_project_by_name(project_name)
                if not project:
                    # 新项目：自动创建并启用监控
                    project = project_mapping.get_or_create_project(project_path, project_name)
                    logger.info(f"Auto-created new project on-the-fly: {project_name} (watcher_enabled={project.watcher_enabled})")
        except Exception as e:
            logger.error(f"Error creating project {project_name}: {e}")
            return
        
        # 检查 watcher_enabled 状态，只有启用时才添加监控
        if not self._get_project_watcher_enabled(project_name):
            logger.info(f"New project created but watcher_enabled=false: {project_name}")
            return
        
        # 创建新的处理器
        handler = FileChangeHandler(
            watch_root=project_path,
            project_name=project_name,
            db_session_factory=get_db_session,
            debounce_interval=self.debounce_interval,
        )
        
        self._project_handlers[project_name] = handler
        
        # 添加到 observer
        if self._observer:
            self._observer.schedule(handler, str(project_path), recursive=True)
        
        # 更新状态
        self._status.watched_projects = list(self._project_handlers.keys())
        
        logger.info(f"Added new project to watch: {project_name}")
    
    def _handle_project_deleted(self, project_path: Path) -> None:
        """处理项目删除"""
        if not self._project_handler:
            return
        
        if not self._project_handler._is_valid_project_dir(project_path):
            return
        
        project_name = project_path.name
        
        # 移除处理器
        if project_name in self._project_handlers:
            del self._project_handlers[project_name]
        
        # 删除 RAG 项目
        if self._project_handler:
            self._project_handler.on_project_deleted(project_path)
        
        # 更新状态
        self._status.watched_projects = list(self._project_handlers.keys())
        
        logger.info(f"Removed project from watch: {project_name}")
    
    def _handle_project_moved(self, src_path: Path, dest_path: Path) -> None:
        """处理项目重命名"""
        if not self._project_handler:
            return
        
        if not self._project_handler._is_valid_project_dir(src_path):
            return
        
        old_name = src_path.name
        new_name = dest_path.name
        
        # 检查新项目是否启用了监控
        new_watcher_enabled = self._get_project_watcher_enabled(new_name)
        
        # 更新处理器
        if old_name in self._project_handlers:
            if new_watcher_enabled:
                # 继续监控，更新处理器信息
                handler = self._project_handlers.pop(old_name)
                handler.project_name = new_name
                handler.watch_root = dest_path
                self._project_handlers[new_name] = handler
                logger.info(f"Renamed project and continue watching: {old_name} -> {new_name}")
            else:
                # 新项目 watcher_enabled=false，停止监控
                del self._project_handlers[old_name]
                logger.info(f"Renamed project but stop watching (watcher_enabled=false): {old_name} -> {new_name}")
        
        # 更新 RAG 项目
        if self._project_handler:
            self._project_handler.on_project_moved(src_path, dest_path)
        
        # 更新状态
        self._status.watched_projects = list(self._project_handlers.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取监控状态
        
        Returns:
            状态信息字典
        """
        return self._status.to_dict()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取同步统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "global": {
                "total_projects": len(self._project_handlers),
                "total_created": 0,
                "total_updated": 0,
                "total_deleted": 0,
                "total_renamed": 0,
                "total_skipped": 0,
                "total_errors": 0,
            },
            "projects": {},
        }
        
        for project_name, handler in self._project_handlers.items():
            handler_stats = handler.get_stats()
            stats["projects"][project_name] = handler_stats
            
            # 累加全局统计
            stats["global"]["total_created"] += handler.stats.created
            stats["global"]["total_updated"] += handler.stats.updated
            stats["global"]["total_deleted"] += handler.stats.deleted
            stats["global"]["total_renamed"] += handler.stats.renamed
            stats["global"]["total_skipped"] += handler.stats.skipped
            stats["global"]["total_errors"] += handler.stats.errors
        
        return stats
    
    def reset_stats(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        重置统计信息
        
        Args:
            project_name: 指定项目名称，None 表示重置所有
            
        Returns:
            操作结果
        """
        if project_name:
            if project_name in self._project_handlers:
                self._project_handlers[project_name].reset_stats()
                return {
                    "success": True,
                    "message": f"Stats reset for project: {project_name}",
                }
            else:
                return {
                    "success": False,
                    "message": f"Project not found: {project_name}",
                }
        else:
            for handler in self._project_handlers.values():
                handler.reset_stats()
            return {
                "success": True,
                "message": "Stats reset for all projects",
            }
    
    def force_scan(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """
        强制扫描项目（后台异步执行）
        
        Args:
            project_name: 指定项目名称，None 表示扫描所有
            
        Returns:
            操作结果（立即返回，扫描在后台执行）
        """
        if not self._status.is_running:
            return {
                "success": False,
                "message": "Watcher is not running",
            }
        
        # 在后台线程执行扫描
        scan_thread = threading.Thread(
            target=self._force_scan_internal,
            args=(project_name,),
            daemon=True
        )
        scan_thread.start()
        
        return {
            "success": True,
            "message": f"Scanning started for {'project: ' + project_name if project_name else 'all projects'} (running in background)",
        }
    
    def _force_scan_internal(self, project_name: Optional[str] = None) -> None:
        """内部扫描实现（在后台线程执行）"""
        scanned = []
        errors = []
        
        if project_name:
            if project_name in self._project_handlers:
                try:
                    handler = self._project_handlers[project_name]
                    handler._scan_directory(handler.watch_root)
                    scanned.append(project_name)
                except Exception as e:
                    errors.append(f"{project_name}: {str(e)}")
                    logger.error(f"Error scanning project {project_name}: {e}")
            else:
                logger.warning(f"Project not found for scan: {project_name}")
                return
        else:
            for name, handler in list(self._project_handlers.items()):
                try:
                    handler._scan_directory(handler.watch_root)
                    scanned.append(name)
                except Exception as e:
                    errors.append(f"{name}: {str(e)}")
                    logger.error(f"Error scanning project {name}: {e}")
        
        if errors:
            logger.warning(f"Scan completed with {len(errors)} errors")
        else:
            logger.info(f"Scan completed for {len(scanned)} projects")
    
    def start_periodic_checks(self) -> None:
        """启动定期检查线程"""
        # 初始化停止事件
        self._stop_event = threading.Event()
        
        # 一致性检查线程
        self._consistency_check_thread = threading.Thread(
            target=self._consistency_check_loop,
            daemon=True
        )
        self._consistency_check_thread.start()
        logger.info("Started periodic consistency check thread")
        
        # 健康检查线程
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        logger.info("Started periodic health check thread")
    
    def stop_periodic_checks(self) -> None:
        """停止定期检查线程"""
        if self._stop_event:
            self._stop_event.set()
        
        if self._consistency_check_thread:
            self._consistency_check_thread.join(timeout=5)
            self._consistency_check_thread = None
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
            self._health_check_thread = None
        
        if self._stop_event:
            self._stop_event = None
        logger.info("Stopped periodic check threads")
    
    def _consistency_check_loop(self) -> None:
        """定期一致性检查循环"""
        while not self._stop_event.is_set() and self._status.is_running:
            # 等待间隔时间或停止信号
            if self._stop_event.wait(timeout=CONSISTENCY_CHECK_INTERVAL):
                break
            
            if not self._status.is_running:
                break
            
            logger.info("Starting periodic consistency check...")
            try:
                self._force_scan_internal()
                logger.info("Periodic consistency check completed")
            except Exception as e:
                logger.error(f"Error in periodic consistency check: {e}")
    
    def _health_check_loop(self) -> None:
        """定期健康检查循环"""
        while not self._stop_event.is_set() and self._status.is_running:
            # 等待间隔时间或停止信号
            if self._stop_event.wait(timeout=HEALTH_CHECK_INTERVAL):
                break
            
            if not self._status.is_running:
                break
            
            try:
                self._check_qdrant_health()
                self._check_ollama_health()
            except Exception as e:
                logger.error(f"Error in health check: {e}")
    
    def _check_qdrant_health(self) -> None:
        """检查 Qdrant 健康"""
        try:
            from src.core.vector_store import VectorStore
            from src.rag_api.models.database import Project, Chunk
            vector_store = VectorStore()
            
            # 检查每个项目的向量索引状态
            for project_name, handler in list(self._project_handlers.items()):
                try:
                    with get_db_session() as db:
                        project = db.query(Project).filter(
                            Project.name == project_name
                        ).first()
                        if not project:
                            continue
                        
                        project_id = str(project.id)
                        collection_name = f"project_{project_id}"
                        info = vector_store.client.get_collection(collection_name)
                        
                        # 如果有向量但没有索引，发出警告
                        if info.points_count > 100 and info.indexed_vectors_count == 0:
                            logger.warning(
                                f"⚠️ 项目 {project_name} 有 {info.points_count} 个向量但未索引！"
                                f"搜索性能可能受影响。"
                            )
                        
                        # 检查向量数量一致性
                        db_chunk_count = db.query(Chunk).filter(
                            Chunk.project_id == project_id
                        ).count()
                        qdrant_count = info.points_count
                        
                        # 允许 5% 的差异（因为索引延迟）
                        if abs(db_chunk_count - qdrant_count) > max(db_chunk_count, qdrant_count) * 0.05:
                            logger.warning(
                                f"⚠️ 项目 {project_name} 向量数量不一致: "
                                f"数据库 {db_chunk_count} vs Qdrant {qdrant_count}"
                            )
                        
                except Exception as e:
                    logger.warning(f"检查项目 {project_name} Qdrant 状态失败: {e}")
                    
        except Exception as e:
            logger.error(f"Qdrant 健康检查失败: {e}")
    
    def _check_ollama_health(self) -> None:
        """检查 Ollama 健康"""
        try:
            import httpx
            from src.rag_api.config import get_settings
            settings = get_settings()
            ollama_url = f"http://{settings.OLLAMA_HOST.replace('http://', '').replace('https://', '')}/api/tags"
            response = httpx.get(ollama_url, timeout=5)
            if response.status_code != 200:
                logger.warning(f"⚠️ Ollama 服务异常: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama 服务不可用: {e}")


# 全局管理器实例
def get_watcher_manager(projects_root: Optional[Path] = None) -> WatcherManager:
    """
    获取监控管理器实例
    
    Args:
        projects_root: 项目根目录
        
    Returns:
        WatcherManager 实例
    """
    return WatcherManager(projects_root)
