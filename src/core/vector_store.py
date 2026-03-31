"""向量数据库封装 - Qdrant"""

import logging
from typing import Any, Dict, List, Optional

import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, VectorParams

from src.rag_api.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class VectorStore:
    """向量数据库封装"""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
            timeout=settings.QDRANT_TIMEOUT,
            check_compatibility=False,  # 禁用版本检查
        )
        self.vector_size = settings.OLLAMA_EMBED_DIM
        self.collection_prefix = settings.QDRANT_COLLECTION_PREFIX

    def _get_collection_name(self, project_id: str) -> str:
        """获取项目对应的 Collection 名称"""
        return f"{self.collection_prefix}{project_id}"

    def create_collection(self, project_id: str) -> bool:
        """创建项目的 Collection"""
        collection_name = self._get_collection_name(project_id)

        try:
            # 检查是否已存在
            self.client.get_collection(collection_name)
            logger.debug(f"Collection already exists: {collection_name}")
            return True
        except UnexpectedResponse:
            # 不存在则创建
            pass

        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"创建 Collection 失败: {e}")
            return False

    def delete_collection(self, project_id: str) -> bool:
        """删除项目的 Collection"""
        collection_name = self._get_collection_name(project_id)

        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除 Collection 失败: {e}")
            return False

    def add_vector(
        self,
        project_id: str,
        vector: List[float],
        payload: Dict[str, Any],
        vector_id: Optional[str] = None,
    ) -> Optional[str]:
        """添加单个向量"""
        collection_name = self._get_collection_name(project_id)
        point_id = vector_id or str(uuid.uuid4())

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            return point_id
        except Exception as e:
            logger.error(f"添加向量失败: {e}")
            return None

    def add_vectors_batch(
        self,
        project_id: str,
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ) -> List[Optional[str]]:
        """批量添加向量
        
        Returns:
            与输入等长的 ID 列表，失败位置返回 None
            
        Raises:
            ValueError: 批量添加失败时抛出异常（不返回空列表）
        """
        if not vectors or not payloads:
            return []
        
        if len(vectors) != len(payloads):
            raise ValueError("vectors 和 payloads 长度不一致")
        
        collection_name = self._get_collection_name(project_id)
        
        # 尝试批量添加
        try:
            point_ids = [str(uuid.uuid4()) for _ in vectors]
            points = [
                models.PointStruct(
                    id=point_id,
                    vector=vec,
                    payload=payload,
                )
                for point_id, vec, payload in zip(point_ids, vectors, payloads)
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points,
            )
            return point_ids
        except Exception as e:
            logger.error(f"批量添加向量失败: {e}")
            # ⚠️ 不返回空列表！抛出异常让调用方处理
            # 调用方（document_service）会保留 chunks，可后续重新处理
            raise RuntimeError(f"批量添加向量失败: {e}")
            return []

    def search(
        self,
        project_id: str,
        vector: List[float],
        top_k: int = 20,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[models.ScoredPoint]:
        """向量搜索"""
        collection_name = self._get_collection_name(project_id)

        try:
            results = self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=self._build_filter(filters) if filters else None,
            )
            return results.points if hasattr(results, 'points') else []
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []

    def delete_vector(self, project_id: str, vector_id: str) -> bool:
        """删除向量"""
        collection_name = self._get_collection_name(project_id)

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=[vector_id],
                ),
            )
            return True
        except Exception as e:
            logger.error(f"删除向量失败: {e}")
            return False

    def count_vectors(self, project_id: str) -> int:
        """统计向量数量"""
        collection_name = self._get_collection_name(project_id)

        try:
            result = self.client.count(collection_name=collection_name)
            return result.count
        except Exception as e:
            logger.error(f"统计向量数量失败: {e}")
            return 0

    def _build_filter(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
        """构建过滤条件"""
        if not filters:
            return None

        conditions = []

        for key, value in filters.items():
            if isinstance(value, list):
                # 多值匹配
                conditions.append(
                    models.FieldCondition(
                        key=f"payload.{key}",
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                # 单值匹配
                conditions.append(
                    models.FieldCondition(
                        key=f"payload.{key}",
                        match=models.MatchValue(value=value),
                    )
                )

        return models.Filter(must=conditions)

    def collection_exists(self, project_id: str) -> bool:
        """检查 Collection 是否存在"""
        collection_name = self._get_collection_name(project_id)

        try:
            self.client.get_collection(collection_name)
            return True
        except:
            return False
    
    def get_all_point_ids(self, project_id: str) -> List[str]:
        """获取Collection中所有向量的ID"""
        collection_name = self._get_collection_name(project_id)
        point_ids = []
        
        try:
            # 使用 scroll API 获取所有点
            offset = None
            while True:
                result = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_vectors=False,
                )
                
                # scroll 返回 (points, next_page_offset) 元组
                points, next_offset = result if isinstance(result, tuple) else (result, None)
                
                for point in points:
                    point_ids.append(str(point.id))
                
                offset = next_offset
                if offset is None:
                    break
                    
        except Exception as e:
            logger.error(f"获取向量ID列表失败: {e}")
        
        return point_ids
    
    def delete_vectors_batch(self, project_id: str, vector_ids: List[str]) -> int:
        """批量删除向量"""
        if not vector_ids:
            return 0
        
        collection_name = self._get_collection_name(project_id)
        
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(
                    points=vector_ids,
                ),
            )
            return len(vector_ids)
        except Exception as e:
            logger.error(f"批量删除向量失败: {e}")
            return 0