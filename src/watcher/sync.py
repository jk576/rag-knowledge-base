"""项目同步逻辑

处理文件夹与 RAG 项目的映射关系，以及文件的同步操作。
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import UploadFile
from io import BytesIO
from sqlalchemy.orm import Session

from src.rag_api.config import get_settings
from src.rag_api.models.database import Chunk as ChunkModel
from src.rag_api.models.database import Document as DocumentModel
from src.rag_api.models.database import Project as ProjectModel
from src.core.chunker import TextChunker
from src.core.document_processor import DocumentProcessor
from src.core.embedding import EmbeddingService
from src.core.vector_store import VectorStore
from src.services.document_service import DocumentService
# 从统一配置导入文件类型定义
from src.core.comment_extractor import (
    DOC_EXTENSIONS,
    IMAGE_EXTENSIONS,
    CODE_EXTENSIONS,
    CONFIG_EXTENSIONS,  # 配置文件类型（不入索引）
)

# 支持的文件扩展名（不包括代码文件和配置文件）
SUPPORTED_EXTENSIONS = DOC_EXTENSIONS | IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)
settings = get_settings()


class ProjectMapping:
    """文件夹到项目的映射管理"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_project(self, folder_path: Path, folder_name: str) -> ProjectModel:
        """
        获取或创建项目
        
        根据文件夹名查找对应的项目，如果不存在则创建新项目。
        
        Args:
            folder_path: 文件夹完整路径
            folder_name: 文件夹名（作为项目名）
            
        Returns:
            ProjectModel 实例
        """
        # 先查找是否存在同名项目
        project = self.db.query(ProjectModel).filter(
            ProjectModel.name == folder_name
        ).first()
        
        if project:
            logger.debug(f"Found existing project '{folder_name}' with ID {project.id}")
            return project
        
        # 创建新项目
        project = ProjectModel(
            id=str(uuid4()),
            name=folder_name,
            folder_path=str(folder_path),  # 保存项目源文件夹路径
            description=f"Auto-synced from {folder_path}",
            document_count=0,
            chunk_count=0,
            watcher_enabled=0,  # 自动创建的项目默认关闭同步
        )
        self.db.add(project)
        self.db.commit()
        self.db.refresh(project)
        
        # 创建 Qdrant Collection
        vector_store = VectorStore()
        vector_store.create_collection(str(project.id))
        
        logger.info(f"Created new project '{folder_name}' with ID {project.id}")
        return project
    
    def get_project_by_name(self, folder_name: str) -> Optional[ProjectModel]:
        """
        根据文件夹名获取项目
        
        Args:
            folder_name: 文件夹名
            
        Returns:
            ProjectModel 实例或 None
        """
        return self.db.query(ProjectModel).filter(
            ProjectModel.name == folder_name
        ).first()
    
    def update_project_name(self, old_name: str, new_name: str) -> Optional[ProjectModel]:
        """
        更新项目名称（目录重命名时调用）
        
        Args:
            old_name: 原文件夹名
            new_name: 新文件夹名
            
        Returns:
            更新后的 ProjectModel 实例或 None
        """
        project = self.db.query(ProjectModel).filter(
            ProjectModel.name == old_name
        ).first()
        
        if not project:
            logger.warning(f"Project with name '{old_name}' not found for renaming")
            return None
        
        # 检查新名称是否已存在
        existing = self.db.query(ProjectModel).filter(
            ProjectModel.name == new_name
        ).first()
        
        if existing and existing.id != project.id:
            logger.error(f"Cannot rename project: name '{new_name}' already exists")
            return None
        
        project.name = new_name
        project.description = f"Auto-synced (renamed from {old_name})"
        project.updated_at = datetime.utcnow()
        self.db.commit()
        self.db.refresh(project)
        
        logger.info(f"Renamed project from '{old_name}' to '{new_name}'")
        return project
    
    def delete_project_by_name(self, folder_name: str) -> bool:
        """
        删除项目及其所有数据
        
        Args:
            folder_name: 文件夹名
            
        Returns:
            是否成功删除
        """
        project = self.db.query(ProjectModel).filter(
            ProjectModel.name == folder_name
        ).first()
        
        if not project:
            logger.warning(f"Project with name '{folder_name}' not found for deletion")
            return False
        
        project_id = project.id
        
        # 删除 Qdrant Collection
        vector_store = VectorStore()
        vector_store.delete_collection(str(project_id))
        
        # 删除项目文件目录
        project_dir = settings.PROJECTS_DIR / str(project_id)
        if project_dir.exists():
            shutil.rmtree(project_dir)
            logger.debug(f"Removed project directory: {project_dir}")
        
        # 删除数据库记录（级联删除文档和分块）
        self.db.delete(project)
        self.db.commit()
        
        logger.info(f"Deleted project '{folder_name}' (ID: {project_id})")
        return True


class ConsistencyChecker:
    """一致性检查器
    
    检查并修复 RAG 项目目录与原目录之间的不一致。
    """
    
    def __init__(self, db: Session, project_id: str, watch_root: Path):
        self.db = db
        self.project_id = project_id
        self.watch_root = Path(watch_root)
        self.project_dir = settings.PROJECTS_DIR / project_id
        self.vector_store = VectorStore()
        self.stats = {"orphaned_files": 0, "missing_files": 0, "untracked_files": 0, "cleaned": 0, "orphan_vectors": 0}
    
    def check_and_fix(self) -> Dict[str, Any]:
        """
        执行一致性检查并修复
        
        检查项：
        1. RAG中有但原目录已删除的文件（孤儿文件）
        2. 原目录中有但RAG中缺失的文件
        3. RAG项目目录中存在但数据库无记录的文件（未跟踪文件）
        4. Qdrant中有但数据库无记录的向量（孤儿向量）
        
        Returns:
            检查结果统计
        """
        logger.info(f"开始一致性检查: 项目 {self.project_id}")
        
        # 1. 检查孤儿文件（数据库有记录但原目录已删除）
        self._check_orphaned_files()
        
        # 2. 检查缺失文件（原目录有但数据库无记录）
        self._check_missing_files()
        
        # 3. 检查未跟踪文件（RAG项目目录有但数据库无记录）
        self._check_untracked_files()
        
        # 4. 检查孤儿向量（Qdrant有向量但数据库无chunk记录）
        self._check_orphan_vectors()
        
        logger.info(
            f"一致性检查完成: 孤儿文件 {self.stats['orphaned_files']}, "
            f"孤儿向量 {self.stats['orphan_vectors']}, "
            f"未跟踪文件 {self.stats['untracked_files']}, 已清理 {self.stats['cleaned']}"
        )
        return self.stats
    
    def _check_orphaned_files(self):
        """检查并清理孤儿文件（数据库有记录但源文件已删除）"""
        # 获取数据库中所有文档
        docs = self.db.query(DocumentModel).filter(
            DocumentModel.project_id == self.project_id
        ).all()
        
        orphaned_count = 0
        for doc in docs:
            # 检查原文件是否存在
            source_path = self.watch_root / doc.filename
            if not source_path.exists():
                # 这是孤儿文件，需要清理
                logger.warning(f"发现孤儿文件: {doc.filename}")
                orphaned_count += 1
                self._cleanup_orphaned_document(doc)
        
        if orphaned_count > 0:
            logger.info(f"一致性检查: 清理了 {orphaned_count} 个孤儿文件")
        
        self.stats["orphaned_files"] = orphaned_count
    
    def _cleanup_orphaned_document(self, doc: DocumentModel):
        """清理单个孤儿文档"""
        try:
            chunk_count = doc.chunk_count
            
            # 删除向量
            chunks = self.db.query(ChunkModel).filter(
                ChunkModel.document_id == doc.id
            ).all()
            
            for chunk in chunks:
                if chunk.vector_id:
                    try:
                        self.vector_store.delete_vector(self.project_id, chunk.vector_id)
                    except Exception as e:
                        logger.error(f"删除向量失败: {e}")
                self.db.delete(chunk)
            
            # 删除RAG目录中的物理文件
            rag_file_path = self.project_dir / doc.filename
            if rag_file_path.exists():
                rag_file_path.unlink()
                self._cleanup_empty_dirs(rag_file_path.parent)
            
            # 删除数据库记录
            self.db.delete(doc)
            self.db.commit()
            
            # 更新项目统计
            project = self.db.query(ProjectModel).filter(
                ProjectModel.id == self.project_id
            ).first()
            if project:
                project.document_count = max(0, project.document_count - 1)
                project.chunk_count = max(0, project.chunk_count - chunk_count)
                self.db.commit()
            
            self.stats["cleaned"] += 1
            logger.info(f"已清理孤儿文件: {doc.filename}")
            
        except Exception as e:
            logger.error(f"清理孤儿文件失败 {doc.filename}: {e}")
            self.db.rollback()
    
    def _cleanup_empty_dirs(self, dir_path: Path):
        """递归清理空目录"""
        try:
            if dir_path == self.project_dir:
                return  # 不删除项目根目录
            
            if dir_path.exists() and not any(dir_path.iterdir()):
                dir_path.rmdir()
                logger.debug(f"删除空目录: {dir_path}")
                # 递归检查父目录
                self._cleanup_empty_dirs(dir_path.parent)
        except OSError:
            pass  # 目录不为空或无法删除
    
    def _check_missing_files(self):
        """检查缺失的文件（原目录有但RAG中没有）
        
        注意：只统计支持的文件类型（文档和图片）
        代码和配置文件不入索引，不统计为 missing
        """
        if not self.watch_root.exists():
            return
        
        # 获取RAG中所有文件名
        rag_files = set()
        docs = self.db.query(DocumentModel).filter(
            DocumentModel.project_id == self.project_id
        ).all()
        for doc in docs:
            rag_files.add(doc.filename)
        
        # 扫描原目录（只统计支持的文件类型）
        for file_path in self.watch_root.rglob("*"):
            if file_path.is_file():
                # 检查是否是支持的文件类型
                ext = file_path.suffix.lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue  # 跳过代码、配置等不支持的文件
                
                rel_path = str(file_path.relative_to(self.watch_root))
                if rel_path not in rag_files:
                    self.stats["missing_files"] += 1
                    logger.info(f"发现缺失文件（将在同步时添加）: {rel_path}")
    
    def _check_untracked_files(self):
        """
        检查未跟踪的物理文件
        
        扫描 RAG 项目目录，找出存在物理文件但数据库无记录的文件。
        这种情况可能发生在：
        1. 数据库记录被手动删除但物理文件残留
        2. 文件被错误复制到错误的项目目录
        3. 同步过程中出现错误导致记录未写入
        
        注意：为安全起见，只记录警告，不自动删除文件。
        """
        if not self.project_dir.exists():
            return
        
        # 获取数据库中所有文件名（相对路径）
        db_files = set()
        docs = self.db.query(DocumentModel).filter(
            DocumentModel.project_id == self.project_id
        ).all()
        for doc in docs:
            db_files.add(doc.filename)
        
        # 扫描 RAG 项目目录中的所有文件
        untracked_files = []
        for file_path in self.project_dir.rglob("*"):
            if file_path.is_file():
                # 计算相对项目目录的路径
                try:
                    rel_path = str(file_path.relative_to(self.project_dir))
                except ValueError:
                    continue
                
                # 如果数据库中没有这个文件的记录，就是未跟踪文件
                if rel_path not in db_files:
                    untracked_files.append(rel_path)
                    self.stats["untracked_files"] += 1
        
        # 只记录警告，不自动删除（避免误删）
        if untracked_files:
            logger.warning(
                f"发现 {len(untracked_files)} 个未跟踪的文件，建议手动检查或运行一致性检查：\n"
                + "\n".join(f"  - {f}" for f in untracked_files[:10])
                + ("\n  ..." if len(untracked_files) > 10 else "")
            )
    
    def _check_orphan_vectors(self):
        """
        检查并清理孤儿向量（Qdrant中有向量但数据库无chunk记录）
        
        这可能发生在：
        1. Watcher停止期间删除了文件但一致性检查未正确清理
        2. 向量添加成功但数据库写入失败
        3. 数据库被手动清空但Qdrant未同步
        """
        try:
            # 获取数据库中所有chunk的vector_id
            db_vector_ids = set()
            chunks = self.db.query(ChunkModel).filter(
                ChunkModel.project_id == self.project_id
            ).all()
            for chunk in chunks:
                if chunk.vector_id:
                    db_vector_ids.add(chunk.vector_id)
            
            # 获取Qdrant中所有向量的ID
            qdrant_point_ids = self.vector_store.get_all_point_ids(self.project_id)
            
            # 找出孤儿向量（在Qdrant中但不在数据库中）
            orphan_vector_ids = [vid for vid in qdrant_point_ids if vid not in db_vector_ids]
            
            if orphan_vector_ids:
                logger.warning(f"发现 {len(orphan_vector_ids)} 个孤儿向量，将清理...")
                
                # 批量删除孤儿向量
                deleted_count = self.vector_store.delete_vectors_batch(
                    self.project_id, 
                    orphan_vector_ids
                )
                
                logger.info(f"已清理 {deleted_count} 个孤儿向量")
                self.stats["orphan_vectors"] = len(orphan_vector_ids)
                self.stats["cleaned"] += deleted_count
            else:
                self.stats["orphan_vectors"] = 0
                
        except Exception as e:
            logger.error(f"检查孤儿向量失败: {e}")

class FileSync:
    """文件同步器
    
    处理文件的同步操作：添加、更新、删除。
    实际的处理逻辑委托给 DocumentService。
    """
    
    def __init__(self, db: Session, project_id: str):
        self.db = db
        self.project_id = project_id
        self.doc_service = DocumentService(db)
        self.vector_store = VectorStore()
    
    def is_supported_file(self, file_path: Path) -> bool:
        """检查文件是否是支持的格式（使用统一配置）"""
        ext = file_path.suffix.lower()
        return ext in SUPPORTED_EXTENSIONS
    
    def get_doc_type(self, file_path: Path) -> str:
        """获取文档类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档类型: 'pdf' | 'docx' | 'xlsx' | 'pptx' | 'image' | 'md' | 'txt'
            
        注意：代码文件不入 RAG 索引，此方法不会返回 'code' 类型
        """
        ext = file_path.suffix.lower()
        
        # 文档类型（使用统一配置）
        if ext in DOC_EXTENSIONS:
            doc_type_map = {
                ".pdf": "pdf",
                ".docx": "docx", ".doc": "docx",
                ".xlsx": "xlsx", ".xls": "xlsx",
                ".pptx": "pptx", ".ppt": "pptx",
                ".md": "md", ".txt": "txt", ".rst": "txt",
            }
            return doc_type_map.get(ext, "txt")
        
        # 图片类型（使用统一配置）
        if ext in IMAGE_EXTENSIONS:
            return "image"
        
        return "other"
    
    def get_document_by_filename(self, filename: str) -> Optional[DocumentModel]:
        """根据文件名获取文档"""
        return self.db.query(DocumentModel).filter(
            DocumentModel.project_id == self.project_id,
            DocumentModel.filename == filename
        ).first()
    
    def sync_file(self, source_path: Path, relative_path: str) -> Dict[str, Any]:
        """
        同步文件到 RAG（同步版本）
        
        如果文件已存在则更新，否则创建新文档。
        
        Args:
            source_path: 源文件完整路径
            relative_path: 相对项目根目录的路径（用作 filename）
            
        Returns:
            操作结果字典
        """
        if not self.is_supported_file(source_path):
            return {"status": "skipped", "reason": "unsupported_format"}
        
        # 检查文件是否已存在
        existing_doc = self.get_document_by_filename(relative_path)
        
        if existing_doc:
            # 检查文件是否有变更
            current_mtime = source_path.stat().st_mtime
            doc_mtime = existing_doc.updated_at.timestamp() if existing_doc.updated_at else 0
            
            if current_mtime <= doc_mtime:
                logger.debug(f"File {relative_path} unchanged, skipping")
                return {"status": "skipped", "reason": "unchanged", "doc_id": existing_doc.id}
            
            # 更新现有文档
            return self._update_document(existing_doc, source_path)
        else:
            # 创建新文档
            return self._create_document(source_path, relative_path)
    
    # 保留异步版本以兼容旧代码
    async def sync_file_async(self, source_path: Path, relative_path: str) -> Dict[str, Any]:
        """异步版本的sync_file，用于向后兼容"""
        return self.sync_file(source_path, relative_path)
    
    def _create_document(self, source_path: Path, relative_path: str) -> Dict[str, Any]:
        """创建新文档（同步）"""
        try:
            # 确保项目目录存在
            project_dir = settings.PROJECTS_DIR / self.project_id
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件到项目目录
            dest_path = project_dir / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            
            doc_type = self.get_doc_type(source_path)
            
            # 使用 DocumentService 处理文档
            result = self.doc_service.process_document(
                file_path=dest_path,
                doc_type=doc_type,
                project_id=self.project_id,
                filename=relative_path,
                source_path=str(source_path),  # 传递原始文件路径
                metadata={"source_path": str(source_path)}
            )
            
            if result.success:
                logger.info(f"Created document '{relative_path}' (ID: {result.document_id})")
                return {"status": "created", "doc_id": result.document_id}
            else:
                logger.error(f"Error creating document {relative_path}: {result.error_message}")
                return {"status": "error", "error": result.error_message}
            
        except Exception as e:
            logger.error(f"Error creating document {relative_path}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _update_document(self, doc: DocumentModel, source_path: Path) -> Dict[str, Any]:
        """更新现有文档（同步）"""
        try:
            # 删除旧数据
            old_chunk_count = doc.chunk_count
            
            # 删除旧向量
            chunks = self.db.query(ChunkModel).filter(
                ChunkModel.document_id == doc.id
            ).all()
            
            for chunk in chunks:
                if chunk.vector_id:
                    self.vector_store.delete_vector(self.project_id, chunk.vector_id)
                self.db.delete(chunk)
            
            self.db.commit()
            
            # 复制新文件
            project_dir = settings.PROJECTS_DIR / self.project_id
            dest_path = project_dir / doc.filename
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            
            # 更新文档基本信息
            doc.file_size = dest_path.stat().st_size
            doc.file_path = str(source_path)
            doc.status = "processing"
            doc.updated_at = datetime.utcnow()
            self.db.commit()
            
            # 使用 DocumentService 重新处理
            doc_type = self.get_doc_type(source_path)
            result = self.doc_service.process_document(
                file_path=dest_path,
                doc_type=doc_type,
                project_id=self.project_id,
                document_id=doc.id,
                filename=doc.filename,
                source_path=str(source_path),  # 传递原始文件路径
                metadata={"source_path": str(source_path)}
            )
            
            if result.success:
                logger.info(f"Updated document '{doc.filename}' (ID: {doc.id})")
                return {"status": "updated", "doc_id": doc.id}
            else:
                logger.error(f"Error updating document {doc.filename}: {result.error_message}")
                return {"status": "error", "error": result.error_message}
            
        except Exception as e:
            logger.error(f"Error updating document {doc.filename}: {e}")
            return {"status": "error", "error": str(e)}
    
    def delete_file(self, relative_path: str) -> Dict[str, Any]:
        """
        从 RAG 删除文件
        
        Args:
            relative_path: 相对项目根目录的路径
            
        Returns:
            操作结果字典
        """
        doc = self.get_document_by_filename(relative_path)
        
        if not doc:
            # 数据库中没有记录，但可能物理文件仍存在（孤立文件）
            # 尝试直接删除物理文件
            rag_file_path = settings.PROJECTS_DIR / self.project_id / relative_path
            if rag_file_path.exists():
                try:
                    rag_file_path.unlink()
                    # 清理空目录
                    self._cleanup_empty_dirs(rag_file_path.parent)
                    logger.info(f"Deleted orphaned file without DB record: '{relative_path}'")
                    return {"status": "deleted", "reason": "orphaned_file_no_db_record"}
                except Exception as e:
                    logger.error(f"Error deleting orphaned file {relative_path}: {e}")
                    return {"status": "error", "error": str(e)}
            
            logger.debug(f"Document '{relative_path}' not found for deletion")
            return {"status": "skipped", "reason": "not_found"}
        
        try:
            # 使用 DocumentService 删除
            success = self.doc_service.delete_document(doc.id, delete_file=True)
            
            if success:
                logger.info(f"Deleted document '{relative_path}' (ID: {doc.id})")
                return {"status": "deleted", "doc_id": doc.id}
            else:
                return {"status": "error", "error": "删除失败"}
            
        except Exception as e:
            logger.error(f"Error deleting document {relative_path}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _cleanup_empty_dirs(self, dir_path: Path):
        """递归清理空目录"""
        try:
            project_dir = settings.PROJECTS_DIR / self.project_id
            if dir_path == project_dir:
                return
            
            if dir_path.exists() and not any(dir_path.iterdir()):
                dir_path.rmdir()
                logger.debug(f"删除空目录: {dir_path}")
                self._cleanup_empty_dirs(dir_path.parent)
        except OSError:
            pass
    
    def rename_file(self, old_relative_path: str, new_relative_path: str) -> Dict[str, Any]:
        """
        重命名/移动文件
        
        Args:
            old_relative_path: 原相对路径
            new_relative_path: 新相对路径
            
        Returns:
            操作结果字典
        """
        doc = self.get_document_by_filename(old_relative_path)
        
        if not doc:
            logger.debug(f"Document '{old_relative_path}' not found for renaming")
            return {"status": "skipped", "reason": "not_found"}
        
        try:
            project_dir = settings.PROJECTS_DIR / self.project_id
            
            # 移动文件
            old_path = project_dir / old_relative_path
            new_path = project_dir / new_relative_path
            
            if old_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(old_path), str(new_path))
                
                # 清理旧目录
                try:
                    old_path.parent.rmdir()
                except OSError:
                    pass
            
            # 更新记录
            doc.filename = new_relative_path
            doc.updated_at = datetime.utcnow()
            self.db.commit()
            
            logger.info(f"Renamed document from '{old_relative_path}' to '{new_relative_path}'")
            return {"status": "renamed", "doc_id": doc.id}
            
        except Exception as e:
            logger.error(f"Error renaming document {old_relative_path}: {e}")
            return {"status": "error", "error": str(e)}


class SyncStats:
    """同步统计"""
    
    def __init__(self):
        self.created: int = 0
        self.updated: int = 0
        self.deleted: int = 0
        self.renamed: int = 0
        self.skipped: int = 0
        self.errors: int = 0
        self.last_sync: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "created": self.created,
            "updated": self.updated,
            "deleted": self.deleted,
            "renamed": self.renamed,
            "skipped": self.skipped,
            "errors": self.errors,
            "last_sync": self.last_sync.isoformat() if self.last_sync else None,
        }
    
    def reset(self) -> None:
        """重置统计"""
        self.created = 0
        self.updated = 0
        self.deleted = 0
        self.renamed = 0
        self.skipped = 0
        self.errors = 0
