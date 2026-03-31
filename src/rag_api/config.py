"""配置管理模块"""

import os
import secrets
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import bcrypt
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用信息
    APP_NAME: str = "RAG Knowledge Base"
    APP_VERSION: str = "0.1.0"
    APP_DEBUG: bool = False
    APP_LOG_LEVEL: str = "INFO"
    
    # 服务配置
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    
    # Ollama 配置
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "bge-m3"
    OLLAMA_EMBED_DIM: int = 1024
    OLLAMA_TIMEOUT: int = 60
    
    # Qdrant 配置
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_PREFIX: str = "project_"
    QDRANT_TIMEOUT: int = 30
    
    # 数据目录
    DATA_DIR: Path = Path("./data")
    PROJECTS_DIR: Path = Path("./data/projects")
    VECTOR_DB_DIR: Path = Path("./data/vector_db")
    DB_PATH: Path = Path("./db/metadata.db")
    
    # MinerU 配置
    MINERU_DEVICE: str = "mps"  # cpu, cuda, mps
    MINERU_CONFIG_PATH: Path = Path.home() / ".mineru" / "magic-pdf.json"
    
    # 分块配置
    CHUNK_SIZE: int = 1000           # 目标分块大小
    CHUNK_OVERLAP: int = 100         # 重叠大小
    MAX_CHUNK_SIZE: int = 4000       # 硬性上限（防止 Ollama 超限，古籍内容 token 消耗高）
    MIN_CHUNK_SIZE: int = 300        # 最小分块大小（防止过碎）
    CHUNK_SEPARATORS: List[str] = ["\n\n", "\n", "。", "；", " ", ""]
    
    # 语义分块配置
    USE_SEMANTIC_CHUNKING: bool = True  # 启用启发式语义分块
    
    # 检索配置
    SEARCH_TOP_K: int = 40
    SEARCH_SCORE_THRESHOLD: float = 0.7
    SEARCH_RERANK_TOP_K: int = 20
    
    # MCP 配置
    MCP_SERVER_NAME: str = "rag-knowledge-base"
    MCP_TRANSPORT: str = "stdio"

    # 认证配置
    AUTH_ENABLED: bool = True
    SECRET_KEY: str = ""
    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD_HASH: str = ""
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # pydantic-settings v2 配置
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # 忽略未定义的字段（如 ADMIN_PASSWORD）
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 确保目录存在
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)

        # 初始化认证配置
        self._init_auth_config()

    def _init_auth_config(self) -> None:
        """初始化认证配置"""
        if not self.AUTH_ENABLED:
            return

        # SECRET_KEY: 从环境变量读取，不提供则自动生成并警告
        if not self.SECRET_KEY:
            self.SECRET_KEY = secrets.token_urlsafe(32)
            warnings.warn(
                "SECRET_KEY not set in environment, using auto-generated key. "
                "Please set SECRET_KEY in your .env file for production.",
                RuntimeWarning,
                stacklevel=2,
            )


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
