"""BM25 索引模块

提供基于 BM25 的关键词搜索能力，与向量搜索结合实现混合检索。
"""

import json
import logging
import pickle
import threading
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jieba
from rank_bm25 import BM25Okapi

from src.rag_api.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# 预加载 jieba 词典（避免首次搜索延迟）
jieba.initialize()


class BM25Index:
    """BM25 索引
    
    为每个项目维护一个 BM25 索引，支持：
    - 增量更新（添加/删除文档）
    - 批量操作（性能优化）
    - 持久化存储
    - 中文分词
    - 线程安全
    """
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.corpus: List[str] = []  # 文档内容列表
        self.tokenized_corpus: List[List[str]] = []  # 分词后的文档列表
        self.chunk_ids: List[str] = []  # chunk_id 列表
        self._chunk_id_to_idx: Dict[str, int] = {}  # chunk_id -> 索引映射（加速查找）
        self._dirty: bool = False  # 是否有未保存的变更
        self._pending_rebuild: bool = False  # 是否需要重建索引
        self.bm25: Optional[BM25Okapi] = None
        self._index_path = settings.PROJECTS_DIR / project_id / "bm25_index.pkl"
        self._lock = threading.RLock()  # 线程锁
        
    def tokenize(self, text: str) -> List[str]:
        """分词
        
        支持中英文混合文本。
        """
        # 使用 jieba 分词
        tokens = list(jieba.cut(text))
        # 过滤空字符串和标点
        tokens = [t.strip().lower() for t in tokens if t.strip() and len(t.strip()) > 1]
        return tokens
    
    def add_document(self, chunk_id: str, content: str, defer_rebuild: bool = False) -> None:
        """添加文档到索引
        
        Args:
            chunk_id: 文档 chunk ID
            content: 文档内容
            defer_rebuild: 是否延迟重建索引（批量操作时使用）
        """
        with self._lock:
            # 使用字典查找（O(1)）代替列表查找（O(n)）
            if chunk_id in self._chunk_id_to_idx:
                # 更新现有文档
                idx = self._chunk_id_to_idx[chunk_id]
                self.corpus[idx] = content
                self.tokenized_corpus[idx] = self.tokenize(content)
            else:
                # 添加新文档
                idx = len(self.chunk_ids)
                self.corpus.append(content)
                self.tokenized_corpus.append(self.tokenize(content))
                self.chunk_ids.append(chunk_id)
                self._chunk_id_to_idx[chunk_id] = idx
            
            # 标记为脏
            self._dirty = True
            
            # 重建索引（可延迟）
            if defer_rebuild:
                self._pending_rebuild = True
            else:
                self._rebuild_index()
    
    def add_documents_batch(self, documents: List[Tuple[str, str]]) -> None:
        """批量添加文档（性能优化）
        
        Args:
            documents: [(chunk_id, content), ...] 列表
        """
        with self._lock:
            for chunk_id, content in documents:
                if chunk_id in self._chunk_id_to_idx:
                    idx = self._chunk_id_to_idx[chunk_id]
                    self.corpus[idx] = content
                    self.tokenized_corpus[idx] = self.tokenize(content)
                else:
                    idx = len(self.chunk_ids)
                    self.corpus.append(content)
                    self.tokenized_corpus.append(self.tokenize(content))
                    self.chunk_ids.append(chunk_id)
                    self._chunk_id_to_idx[chunk_id] = idx
            
            self._dirty = True
            self._rebuild_index()  # 批量操作后只重建一次
    
    def remove_document(self, chunk_id: str, defer_rebuild: bool = False) -> bool:
        """从索引中删除文档
        
        Args:
            chunk_id: 要删除的文档 chunk ID
            defer_rebuild: 是否延迟重建索引（批量操作时使用）
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if chunk_id not in self._chunk_id_to_idx:
                return False
            
            idx = self._chunk_id_to_idx[chunk_id]
            
            # 删除元素
            self.corpus.pop(idx)
            self.tokenized_corpus.pop(idx)
            self.chunk_ids.pop(idx)
            
            # 重建映射
            self._chunk_id_to_idx = {
                cid: i for i, cid in enumerate(self.chunk_ids)
            }
            
            # 标记为脏
            self._dirty = True
            
            # 重建索引（可延迟）
            if defer_rebuild:
                self._pending_rebuild = True
            else:
                self._rebuild_index()
            return True
    
    def remove_documents_batch(self, chunk_ids: List[str]) -> int:
        """批量删除文档（性能优化）
        
        Args:
            chunk_ids: 要删除的 chunk_id 列表
            
        Returns:
            成功删除的数量
        """
        with self._lock:
            removed = 0
            indices_to_remove = []
            
            for chunk_id in chunk_ids:
                if chunk_id in self._chunk_id_to_idx:
                    indices_to_remove.append(self._chunk_id_to_idx[chunk_id])
                    removed += 1
            
            # 从后往前删除，避免索引变化
            indices_to_remove.sort(reverse=True)
            for idx in indices_to_remove:
                self.corpus.pop(idx)
                self.tokenized_corpus.pop(idx)
                self.chunk_ids.pop(idx)
            
            # 重建映射
            self._chunk_id_to_idx = {
                cid: i for i, cid in enumerate(self.chunk_ids)
            }
            
            self._dirty = True
            self._rebuild_index()  # 批量操作后只重建一次
            
            return removed
    
    def _rebuild_index(self) -> None:
        """重建 BM25 索引"""
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
    
    def search(
        self, 
        query: str, 
        top_k: int = 20,
        score_threshold: float = 0.0
    ) -> List[Tuple[str, float, str]]:
        """搜索
        
        Args:
            query: 查询字符串
            top_k: 返回结果数量
            score_threshold: 分数阈值
            
        Returns:
            List of (chunk_id, score, content)
        """
        with self._lock:
            if not self.bm25 or not self.corpus:
                return []
            
            # 分词
            query_tokens = self.tokenize(query)
            if not query_tokens:
                return []
            
            # BM25 搜索
            scores = self.bm25.get_scores(query_tokens)
            
            # 排序并过滤
            results = []
            for i, score in enumerate(scores):
                if score >= score_threshold:
                    results.append((self.chunk_ids[i], float(score), self.corpus[i]))
            
            # 按分数降序排序
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results[:top_k]
    
    def save(self) -> bool:
        """保存索引到磁盘（原子写入）"""
        with self._lock:
            # 如果有待重建的索引，先重建再保存
            if self._pending_rebuild:
                self._rebuild_index()
                self._pending_rebuild = False
            
            try:
                self._index_path.parent.mkdir(parents=True, exist_ok=True)
                
                data = {
                    "project_id": self.project_id,
                    "corpus": self.corpus,
                    "tokenized_corpus": self.tokenized_corpus,
                    "chunk_ids": self.chunk_ids,
                }
                
                # 原子写入：先写临时文件，再 rename
                fd, temp_path = tempfile.mkstemp(
                    dir=self._index_path.parent,
                    suffix='.tmp'
                )
                try:
                    with os.fdopen(fd, 'wb') as f:
                        pickle.dump(data, f)
                    
                    # 原子操作：rename 是原子的
                    os.replace(temp_path, self._index_path)
                    self._dirty = False
                    
                    logger.debug(f"BM25 索引已保存: {self._index_path}")
                    return True
                    
                except Exception as e:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    raise e
                
            except Exception as e:
                logger.error(f"保存 BM25 索引失败: {e}")
                return False
    
    def load(self) -> bool:
        """从磁盘加载索引"""
        try:
            if not self._index_path.exists():
                logger.debug(f"BM25 索引文件不存在: {self._index_path}")
                return False
            
            with open(self._index_path, "rb") as f:
                data = pickle.load(f)
            
            self.corpus = data.get("corpus", [])
            self.tokenized_corpus = data.get("tokenized_corpus", [])
            self.chunk_ids = data.get("chunk_ids", [])
            
            # 重建映射
            self._chunk_id_to_idx = {
                cid: i for i, cid in enumerate(self.chunk_ids)
            }
            
            # 重建索引对象
            self._rebuild_index()
            
            self._dirty = False
            
            logger.debug(f"BM25 索引已加载: {len(self.corpus)} 个文档")
            return True
            
        except Exception as e:
            logger.error(f"加载 BM25 索引失败: {e}")
            return False
    
    def clear(self) -> None:
        """清空索引"""
        self.corpus.clear()
        self.tokenized_corpus.clear()
        self.chunk_ids.clear()
        self._chunk_id_to_idx.clear()
        self._dirty = False
        self.bm25 = None
    
    @property
    def doc_count(self) -> int:
        """索引中的文档数量"""
        return len(self.corpus)


class BM25IndexManager:
    """BM25 索引管理器
    
    管理多个项目的 BM25 索引，提供缓存和增量更新。
    线程安全。
    """
    
    def __init__(self):
        self._indexes: Dict[str, BM25Index] = {}
        self._lock = threading.RLock()
    
    def get_index(self, project_id: str) -> BM25Index:
        """获取或创建项目的 BM25 索引"""
        with self._lock:
            if project_id not in self._indexes:
                index = BM25Index(project_id)
                index.load()  # 尝试从磁盘加载
                self._indexes[project_id] = index
            
            return self._indexes[project_id]
    
    def build_index_from_db(self, project_id: str, db_session) -> BM25Index:
        """从数据库构建完整索引
        
        Args:
            project_id: 项目 ID
            db_session: 数据库会话
            
        Returns:
            构建完成的 BM25 索引
        """
        from src.rag_api.models.database import Chunk
        
        with self._lock:
            index = self.get_index(project_id)
            index.clear()
            
            # 从数据库加载所有 chunks
            chunks = db_session.query(Chunk).filter(
                Chunk.project_id == project_id
            ).order_by(Chunk.chunk_index).all()
            
            # 批量添加（性能优化）
            documents = [(chunk.id, chunk.content) for chunk in chunks]
            index.add_documents_batch(documents)
            
            # 保存到磁盘
            index.save()
            
            logger.info(f"BM25 索引构建完成: 项目 {project_id}, {len(chunks)} 个文档")
            return index
    
    def save_all(self) -> None:
        """保存所有索引"""
        with self._lock:
            for index in self._indexes.values():
                index.save()
    
    def clear_cache(self, project_id: Optional[str] = None) -> None:
        """清除缓存
        
        Args:
            project_id: 指定项目 ID，None 则清除全部
        """
        with self._lock:
            if project_id:
                if project_id in self._indexes:
                    del self._indexes[project_id]
            else:
                self._indexes.clear()


# 全局索引管理器
bm25_manager = BM25IndexManager()