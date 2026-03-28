"""层次化索引模块 - 简化版 RAPTOR

实现文档摘要层 + 详细 chunks 层的两阶段检索：
1. 文档摘要：每个文档生成一个摘要向量
2. 详细 chunks：原始文档分块的向量

检索流程：
Query → 摘要搜索（找到相关文档）→ chunks 搜索（在这些文档中找细节）
"""

import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from src.rag_api.config import get_settings
from src.core.embedding import EmbeddingService
from src.core.vector_store import VectorStore

settings = get_settings()
logger = logging.getLogger(__name__)

import uuid


@dataclass
class DocumentSummary:
    """文档摘要"""
    doc_id: str
    project_id: str
    filename: str
    summary: str
    chunk_count: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SummaryGenerator:
    """文档摘要生成器
    
    使用 LLM 生成文档摘要，用于层次化检索。
    """
    
    def __init__(self, model: str = "qwen3:8b"):
        self.model = model
        self._ollama_host = settings.OLLAMA_HOST
    
    async def generate_summary(
        self, 
        chunks: List[str], 
        max_chunks: int = 10,
        max_length: int = 500
    ) -> str:
        """生成文档摘要
        
        Args:
            chunks: 文档分块列表
            max_chunks: 最多使用的 chunk 数量
            max_length: 摘要最大长度（字符）
            
        Returns:
            生成的摘要
        """
        if not chunks:
            return ""
        
        # 取前 N 个 chunks（覆盖文档主要内容的开头部分）
        selected_chunks = chunks[:max_chunks]
        content = "\n\n".join(selected_chunks)
        
        # 截断过长内容
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        prompt = f"""请为以下文档内容生成一个简洁的摘要（{max_length}字以内）。

摘要应包含：
1. 文档主题和核心内容
2. 关键信息点
3. 重要概念或术语

文档内容：
{content}

摘要："""

        try:
            import httpx
            import json
            
            async with httpx.AsyncClient(timeout=60) as client:
                # 使用 /api/chat 或 /api/generate
                # 尝试 /api/chat（更稳定）
                response = await client.post(
                    f"{self._ollama_host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 512,
                        }
                    }
                )
                
                if response.status_code == 404:
                    # 回退到 /api/generate
                    response = await client.post(
                        f"{self._ollama_host}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "num_predict": 512,
                            }
                        }
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # /api/chat 返回格式
                if "message" in data:
                    summary = data["message"]["content"].strip()
                else:
                    # /api/generate 返回格式
                    summary = data.get("response", "").strip()
                
                # 截断到最大长度
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
                
                return summary
                
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            # 回退：返回第一个 chunk 的前 200 字
            return chunks[0][:200] + "..." if chunks else ""
    
    def generate_summary_sync(
        self, 
        chunks: List[str], 
        max_chunks: int = 10,
        max_length: int = 500
    ) -> str:
        """同步版本的摘要生成"""
        if not chunks:
            return ""
        
        selected_chunks = chunks[:max_chunks]
        content = "\n\n".join(selected_chunks)
        
        if len(content) > 4000:
            content = content[:4000] + "..."
        
        prompt = f"""请为以下文档内容生成一个简洁的摘要（{max_length}字以内）。

摘要应包含：
1. 文档主题和核心内容
2. 关键信息点
3. 重要概念或术语

文档内容：
{content}

摘要："""

        try:
            import httpx
            
            with httpx.Client(timeout=60) as client:
                # 尝试 /api/chat（更稳定）
                response = client.post(
                    f"{self._ollama_host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 512,
                        }
                    }
                )
                
                if response.status_code == 404:
                    # 回退到 /api/generate
                    response = client.post(
                        f"{self._ollama_host}/api/generate",
                        json={
                            "model": self.model,
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0.3,
                                "num_predict": 512,
                            }
                        }
                    )
                
                response.raise_for_status()
                data = response.json()
                
                # /api/chat 返回格式
                if "message" in data:
                    summary = data["message"]["content"].strip()
                else:
                    summary = data.get("response", "").strip()
                
                if len(summary) > max_length:
                    summary = summary[:max_length] + "..."
                
                return summary
                
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            return chunks[0][:200] + "..." if chunks else ""


class HierarchicalIndex:
    """层次化索引
    
    管理文档摘要层和 chunks 层的索引。
    """
    
    # 摘要 Collection 名称后缀
    SUMMARY_SUFFIX = "_summaries"
    
    def __init__(self):
        self.embedding = EmbeddingService()
        self.vector_store = VectorStore()
        self.summary_generator = SummaryGenerator()
    
    def _get_summary_collection_name(self, project_id: str) -> str:
        """获取摘要 Collection 名称"""
        return f"project_{project_id}{self.SUMMARY_SUFFIX}"
    
    def create_summary_collection(self, project_id: str) -> bool:
        """创建摘要 Collection"""
        collection_name = self._get_summary_collection_name(project_id)
        
        try:
            # 检查是否已存在
            self.vector_store.client.get_collection(collection_name)
            return True
        except Exception:
            pass
        
        try:
            from qdrant_client.models import Distance, VectorParams
            
            self.vector_store.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_store.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"创建摘要 Collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"创建摘要 Collection 失败: {e}")
            return False
    
    async def index_document(
        self,
        project_id: str,
        document_id: str,
        chunks: List[str],
        filename: str,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """索引文档（生成并存储摘要）
        
        Args:
            project_id: 项目 ID
            document_id: 文档 ID
            chunks: 文档分块列表
            filename: 文件名
            metadata: 额外元数据
            
        Returns:
            摘要向量 ID，失败返回 None
        """
        if not chunks:
            return None
        
        # 确保摘要 Collection 存在
        self.create_summary_collection(project_id)
        
        # 生成摘要
        summary = await self.summary_generator.generate_summary(chunks)
        if not summary:
            logger.warning(f"文档 {document_id} 生成摘要失败")
            return None
        
        # 向量化摘要
        summary_embedding = await self.embedding.embed_text(summary)
        
        # 存储到摘要 Collection
        collection_name = self._get_summary_collection_name(project_id)
        
        payload = {
            "document_id": document_id,
            "filename": filename,
            "summary": summary,
            "chunk_count": len(chunks),
            **(metadata or {})
        }
        
        try:
            from qdrant_client.http import models
            
            # 使用标准 UUID
            vector_id = str(uuid.uuid4())
            
            self.vector_store.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=summary_embedding,
                        payload=payload,
                    )
                ],
            )
            
            logger.info(f"文档摘要已索引: {filename} ({len(chunks)} chunks)")
            return vector_id
            
        except Exception as e:
            logger.error(f"存储文档摘要失败: {e}")
            return None
    
    def index_document_sync(
        self,
        project_id: str,
        document_id: str,
        chunks: List[str],
        filename: str,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """同步版本的文档索引"""
        if not chunks:
            return None
        
        self.create_summary_collection(project_id)
        
        summary = self.summary_generator.generate_summary_sync(chunks)
        if not summary:
            return None
        
        summary_embedding = self.embedding.embed_text_sync(summary)
        collection_name = self._get_summary_collection_name(project_id)
        
        payload = {
            "document_id": document_id,
            "filename": filename,
            "summary": summary,
            "chunk_count": len(chunks),
            **(metadata or {})
        }
        
        try:
            from qdrant_client.http import models
            
            # 使用标准 UUID
            vector_id = str(uuid.uuid4())
            
            self.vector_store.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=vector_id,
                        vector=summary_embedding,
                        payload=payload,
                    )
                ],
            )
            
            logger.info(f"文档摘要已索引: {filename}")
            return vector_id
            
        except Exception as e:
            logger.error(f"存储文档摘要失败: {e}")
            return None
    
    def delete_document_summary(self, project_id: str, document_id: str) -> bool:
        """删除文档摘要"""
        collection_name = self._get_summary_collection_name(project_id)
        
        try:
            self.vector_store.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[f"summary_{document_id}"]
                ),
            )
            return True
        except Exception as e:
            logger.error(f"删除文档摘要失败: {e}")
            return False
    
    def search_summaries(
        self,
        project_id: str,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """搜索文档摘要
        
        Args:
            project_id: 项目 ID
            query: 查询字符串
            top_k: 返回结果数量
            
        Returns:
            摘要结果列表
        """
        collection_name = self._get_summary_collection_name(project_id)
        
        try:
            # 检查 Collection 是否存在
            self.vector_store.client.get_collection(collection_name)
        except Exception:
            return []
        
        # 向量化查询
        query_embedding = self.embedding.embed_text_sync(query)
        
        try:
            results = self.vector_store.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
            )
            
            summaries = []
            for point in results.points:
                summaries.append({
                    "document_id": point.payload.get("document_id"),
                    "filename": point.payload.get("filename"),
                    "summary": point.payload.get("summary"),
                    "chunk_count": point.payload.get("chunk_count"),
                    "score": point.score,
                })
            
            return summaries
            
        except Exception as e:
            logger.error(f"搜索摘要失败: {e}")
            return []


class HierarchicalSearchService:
    """层次化搜索服务
    
    实现两阶段检索：
    1. 摘要搜索：找到相关文档
    2. chunks 搜索：在这些文档中搜索详细内容
    """
    
    def __init__(self):
        self.hierarchical_index = HierarchicalIndex()
        self.vector_store = VectorStore()
        self.embedding = EmbeddingService()
    
    async def search(
        self,
        project_id: str,
        query: str,
        top_k: int = 10,
        summary_top_k: int = 5,
        chunks_per_doc: int = 3
    ) -> Tuple[List[Dict], List[Dict]]:
        """执行层次化搜索
        
        Args:
            project_id: 项目 ID
            query: 查询字符串
            top_k: 最终返回结果数量
            summary_top_k: 摘要搜索返回的文档数
            chunks_per_doc: 每个文档返回的 chunk 数
            
        Returns:
            (相关摘要列表, 详细 chunks 列表)
        """
        # Phase 1: 搜索摘要，找到相关文档
        summaries = self.hierarchical_index.search_summaries(
            project_id=project_id,
            query=query,
            top_k=summary_top_k
        )
        
        if not summaries:
            return [], []
        
        # Phase 2: 在相关文档中搜索 chunks
        relevant_doc_ids = [s["document_id"] for s in summaries]
        
        # 向量化查询
        query_embedding = await self.embedding.embed_text(query)
        
        # 搜索 chunks（过滤只在这些文档中）
        # 注意：Qdrant 的 filter 需要按 document_id 过滤
        chunks = []
        
        for doc_id in relevant_doc_ids:
            try:
                results = self.vector_store.client.query_points(
                    collection_name=f"project_{project_id}",
                    query=query_embedding,
                    limit=chunks_per_doc,
                    query_filter={
                        "must": [
                            {
                                "key": "document_id",
                                "match": {"value": doc_id}
                            }
                        ]
                    }
                )
                
                for point in results.points:
                    chunks.append({
                        "content": point.payload.get("content", ""),
                        "document_id": doc_id,
                        "filename": point.payload.get("filename"),
                        "chunk_id": point.payload.get("chunk_id"),
                        "score": point.score,
                    })
                    
            except Exception as e:
                logger.warning(f"搜索文档 {doc_id} 的 chunks 失败: {e}")
        
        # 按 score 排序并取 top_k
        chunks.sort(key=lambda x: x["score"], reverse=True)
        chunks = chunks[:top_k]
        
        return summaries, chunks


# 全局实例
hierarchical_index = HierarchicalIndex()
hierarchical_search = HierarchicalSearchService()