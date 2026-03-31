"""文档路由"""

import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from src.rag_api.config import get_settings
from src.rag_api.models.database import get_db
from src.rag_api.models.schemas import APIResponse
from src.services.ingest_service import IngestService

router = APIRouter()
settings = get_settings()


@router.post("/{project_id}/documents", response_model=APIResponse)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
    db: Session = Depends(get_db),
):
    """上传单个文档"""
    service = IngestService(db)
    try:
        result = await service.upload_document(project_id, file, metadata)
        return APIResponse(success=True, data=result, message="文档上传成功")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传文档失败: {str(e)}")


@router.post("/{project_id}/documents/batch", response_model=APIResponse)
async def upload_documents_batch(
    project_id: str,
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    """批量上传文档"""
    service = IngestService(db)
    results = []
    errors = []
    
    for file in files:
        try:
            result = await service.upload_document(project_id, file)
            results.append(result)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    
    return APIResponse(
        success=len(errors) == 0,
        data={"success": results, "errors": errors},
        message=f"上传完成: {len(results)} 成功, {len(errors)} 失败",
    )


@router.get("/{project_id}/documents", response_model=APIResponse)
async def list_documents(
    project_id: str,
    skip: int = 0,
    limit: int = 100,
    filename: str = None,
    db: Session = Depends(get_db),
):
    """列出项目下的所有文档
    
    Args:
        project_id: 项目 ID
        skip: 跳过数量（分页偏移，最小 0）
        limit: 返回数量（默认 100，最大 500）
        filename: 文件名搜索（模糊匹配）
    """
    from src.services.project_service import ProjectService
    
    # 参数验证
    if skip < 0:
        skip = 0
    if limit < 1:
        limit = 100
    if limit > 500:
        limit = 500
    
    service = ProjectService(db)
    result = service.list_documents(project_id, skip=skip, limit=limit, filename=filename)
    return APIResponse(success=True, data=result)


@router.get("/{project_id}/documents/{document_id}", response_model=APIResponse)
async def get_document(
    project_id: str,
    document_id: str,
    db: Session = Depends(get_db),
):
    """获取文档详情"""
    from src.services.project_service import ProjectService
    
    service = ProjectService(db)
    document = service.get_document(document_id)
    if not document or document.project_id != project_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    return APIResponse(success=True, data=document)


@router.delete("/{project_id}/documents/{document_id}", response_model=APIResponse)
async def delete_document(
    project_id: str,
    document_id: str,
    db: Session = Depends(get_db),
):
    """删除文档"""
    service = IngestService(db)
    try:
        service.delete_document(project_id, document_id)
        return APIResponse(success=True, message="文档删除成功")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除文档失败: {str(e)}")


@router.post("/{project_id}/documents/{document_id}/reindex", response_model=APIResponse)
async def reindex_document(
    project_id: str,
    document_id: str,
    db: Session = Depends(get_db),
):
    """重新索引文档"""
    service = IngestService(db)
    try:
        result = await service.reindex_document(project_id, document_id)
        return APIResponse(success=True, data=result, message="文档重新索引成功")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重新索引失败: {str(e)}")


@router.get("/{project_id}/documents/{document_id}/export", response_model=APIResponse)
async def export_document(
    project_id: str,
    document_id: str,
    format: str = "txt",  # txt, json
    db: Session = Depends(get_db),
):
    """导出文档解析后的文本内容"""
    from src.services.project_service import ProjectService
    from src.rag_api.models.database import Chunk
    
    service = ProjectService(db)
    document = service.get_document(document_id)
    if not document or document.project_id != project_id:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    # 获取所有文本块
    chunks = db.query(Chunk).filter(
        Chunk.document_id == document_id
    ).order_by(Chunk.chunk_index).all()
    
    if format == "json":
        # JSON 格式
        data = {
            "document_id": document_id,
            "filename": document.filename,
            "doc_type": document.doc_type,
            "file_path": document.file_path,
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "index": c.chunk_index,
                    "content": c.content
                }
                for c in chunks
            ]
        }
        return APIResponse(success=True, data=data)
    else:
        # 纯文本格式
        full_text = "\n\n".join([c.content for c in chunks])
        data = {
            "document_id": document_id,
            "filename": document.filename,
            "doc_type": document.doc_type,
            "file_path": document.file_path,
            "chunk_count": len(chunks),
            "content": full_text
        }
        return APIResponse(success=True, data=data)


@router.get("/{project_id}/export", response_model=APIResponse)
async def export_project(
    project_id: str,
    format: str = "txt",
    db: Session = Depends(get_db),
):
    """导出整个项目的所有文档文本"""
    from src.rag_api.models.database import Document, Chunk
    
    documents = db.query(Document).filter(
        Document.project_id == project_id,
        Document.status == "completed"
    ).all()
    
    if format == "json":
        data = {
            "project_id": project_id,
            "document_count": len(documents),
            "documents": []
        }
        for doc in documents:
            chunks = db.query(Chunk).filter(
                Chunk.document_id == doc.id
            ).order_by(Chunk.chunk_index).all()
            
            data["documents"].append({
                "id": doc.id,
                "filename": doc.filename,
                "file_path": doc.file_path,
                "chunks": [{"index": c.chunk_index, "content": c.content} for c in chunks]
            })
        return APIResponse(success=True, data=data)
    else:
        # 纯文本格式
        lines = [f"# 项目导出\n", f"项目ID: {project_id}\n", f"文档数量: {len(documents)}\n", "="*50 + "\n"]
        
        for doc in documents:
            lines.append(f"\n## 文档: {doc.filename}\n")
            lines.append(f"文件路径: {doc.file_path}\n")
            lines.append("-"*40 + "\n")
            
            chunks = db.query(Chunk).filter(
                Chunk.document_id == doc.id
            ).order_by(Chunk.chunk_index).all()
            
            for chunk in chunks:
                lines.append(f"\n[片段 {chunk.chunk_index}]\n")
                lines.append(chunk.content)
                lines.append("\n")
            
            lines.append("\n" + "="*50 + "\n")
        
        return APIResponse(success=True, data={
            "project_id": project_id,
            "document_count": len(documents),
            "content": "".join(lines)
        })
