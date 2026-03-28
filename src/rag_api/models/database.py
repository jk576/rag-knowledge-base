"""SQLAlchemy 数据库模型"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from src.rag_api.config import get_settings

settings = get_settings()

# 创建引擎
engine = create_engine(
    f"sqlite:///{settings.DB_PATH}",
    connect_args={"check_same_thread": False},
    echo=settings.APP_DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def generate_uuid() -> str:
    """生成UUID"""
    return str(uuid.uuid4())


class Project(Base):
    """项目表"""
    __tablename__ = "projects"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(100), nullable=False, unique=True)
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
    file_path = Column(String(500), nullable=True)  # 原始文件完整路径
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


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
