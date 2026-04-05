"""SQLAlchemy 数据库模型"""

import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker, scoped_session

from src.rag_api.config import get_settings

settings = get_settings()

# 创建引擎 - 添加连接池配置
engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=settings.APP_DEBUG,
    # 连接池配置
    pool_pre_ping=True,  # 连接前ping，自动回收失效连接
    pool_recycle=3600,   # 连接1小时后自动回收
    max_overflow=10,     # 最大溢出连接数
)

# 监听连接事件，设置SQLite优化参数
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """设置SQLite优化参数"""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")  # 启用外键
    cursor.execute("PRAGMA journal_mode=WAL")  # WAL模式，提高并发性能
    cursor.execute("PRAGMA synchronous=NORMAL")  # 同步模式，平衡性能和安全性
    cursor.close()

# 线程安全的Session工厂（用于多线程环境如Watcher）
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
ScopedSession = scoped_session(SessionLocal)

Base = declarative_base()


def generate_uuid() -> str:
    """生成UUID"""
    return str(uuid.uuid4())


class Project(Base):
    """项目表"""
    __tablename__ = "projects"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False, unique=True)
    folder_path = Column(String(500), nullable=True)  # 项目源文件夹路径（用于 Agent read 源文件）
    description = Column(Text, nullable=True)
    document_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    watcher_enabled = Column(Integer, default=0)  # 0=关闭同步（默认）, 1=启用同步
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Document(Base):
    """文档表"""
    __tablename__ = "documents"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    doc_type = Column(String(20), nullable=False)
    file_size = Column(Integer, default=0)
    file_path = Column(String(500), nullable=True)  # 复制后的文件路径（项目目录内）
    source_path = Column(String(500), nullable=True)  # 原始文件完整路径（Agent read 源文件用）
    chunk_count = Column(Integer, default=0)
    status = Column(String(20), default="pending")  # pending, processing, completed, failed
    error_message = Column(Text, nullable=True)
    metadata_json = Column(Text, nullable=True)  # JSON字符串
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Chunk(Base):
    """文档分块表"""
    __tablename__ = "chunks"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    document_id = Column(String(36), nullable=False, index=True)
    project_id = Column(String(36), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    vector_id = Column(String(100), nullable=True)  # Qdrant中的向量ID
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class WatchMapping(Base):
    """文件监控映射表
    
    记录文件夹与 RAG 项目的映射关系，用于文件系统监控。
    """
    __tablename__ = "watch_mappings"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    folder_path = Column(String(500), nullable=False, unique=True, index=True)
    project_id = Column(String(36), nullable=False, index=True)
    folder_name = Column(String(255), nullable=False, index=True)
    is_active = Column(Integer, default=1)  # 0=inactive, 1=active
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_db():
    """初始化数据库"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """获取数据库会话（FastAPI依赖使用）"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话（上下文管理器，推荐用于多线程环境）
    
    使用示例:
        with get_db_session() as db:
            result = db.query(Project).first()
            # 自动commit/rollback
    """
    db = ScopedSession()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
        ScopedSession.remove()  # 清理线程本地存储


def get_db_session_sync() -> Session:
    """获取数据库会话（同步方式，需手动管理）
    
    使用示例:
        db = get_db_session_sync()
        try:
            result = db.query(Project).first()
            db.commit()
        except Exception:
            db.rollback()
        finally:
            db.close()
    """
    return ScopedSession()
