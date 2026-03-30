"""增强版文档处理器 - 集成 Unstructured 进行 Office 文档解析

提供高质量的文档解析，支持多种格式：
- PDF: MinerU (优先) / pypdf (备用)
- Office: Unstructured (docx, xlsx, pptx)
- 图片: pytesseract (OCR)
- 文本: 直接读取
- 代码: CommentExtractor (只提取注释)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union

from src.rag_api.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """文档处理器 - 支持多种格式

    Office 文档 (docx/xlsx/pptx) 优先使用 Unstructured 解析，
    失败时回退到原生解析器。
    
    代码文件只提取注释，不提取代码本身。
    """

    def __init__(self):
        self.mineru_available = self._check_mineru()
        self.unstructured_available = self._check_unstructured()
        self.comment_extractor = self._init_comment_extractor()

        if self.unstructured_available:
            logger.info("✅ Unstructured 解析器已启用")
        if self.mineru_available:
            logger.info("✅ MinerU 解析器已启用")
        if self.comment_extractor:
            logger.info("✅ CommentExtractor 已启用（代码注释提取）")
    
    def _init_comment_extractor(self):
        """初始化代码注释提取器"""
        try:
            from src.core.comment_extractor import CommentExtractor
            return CommentExtractor()
        except ImportError as e:
            logger.warning(f"CommentExtractor 导入失败: {e}")
            return None

    def _check_mineru(self) -> bool:
        """检查 MinerU 是否可用（通过 Python 3.11 子进程）"""
        try:
            import subprocess
            script_path = Path(__file__).parent.parent.parent / "scripts" / "mineru.sh"
            venv_path = Path(__file__).parent.parent.parent / ".venv-311"
            
            # 检查脚本和环境是否存在
            if not script_path.exists() or not venv_path.exists():
                return False
            
            # 尝试运行 --version 或 --help 检测可用性
            # 由于 MinerU 没有 version 参数，检查 Python 3.11 环境中的 magic_pdf
            result = subprocess.run(
                [str(venv_path / "bin" / "python"), "-c", "import magic_pdf"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
            
        except Exception as e:
            logger.debug(f"MinerU 检测失败: {e}")
            return False

    def _check_unstructured(self) -> bool:
        """检查 Unstructured 是否可用"""
        try:
            from unstructured.partition.docx import partition_docx
            return True
        except ImportError:
            return False

    def extract_text(self, file_path: Union[str, Path], doc_type: str) -> str:
        """提取文档文本

        Args:
            file_path: 文件路径
            doc_type: 文档类型 (pdf/docx/xlsx/pptx/image/md/txt/code)

        Returns:
            str: 提取的文本内容

        Raises:
            ValueError: 解析失败时抛出
        """
        file_path = Path(file_path)

        # Office 文档优先使用 Unstructured
        if self.unstructured_available and doc_type in ["docx", "xlsx", "pptx"]:
            try:
                return self._extract_with_unstructured(file_path, doc_type)
            except Exception as e:
                logger.warning(f"Unstructured 解析失败: {e}，回退到原生解析器")
                # 失败时回退到原生解析器
                return self._extract_with_fallback(file_path, doc_type)

        # 其他类型使用原有逻辑
        return self._extract_with_fallback(file_path, doc_type)

    def extract_structured(
        self,
        file_path: Union[str, Path],
        doc_type: str
    ) -> Dict[str, Any]:
        """提取结构化文档（返回完整结构）

        仅支持 Office 文档类型 (docx/xlsx/pptx)。

        Args:
            file_path: 文件路径
            doc_type: 文档类型

        Returns:
            Dict: 包含 text, markdown, tables, sections 等的字典

        Raises:
            ValueError: Unstructured 未安装或不支持的文档类型
        """
        file_path = Path(file_path)

        if not self.unstructured_available:
            raise ValueError("Unstructured 未安装，无法提取结构化数据")

        if doc_type not in ["docx", "xlsx", "pptx"]:
            raise ValueError(f"不支持的文档类型: {doc_type}，仅支持 docx/xlsx/pptx")

        from src.core.unstructured_parser import UnstructuredOfficeParser

        parser = UnstructuredOfficeParser()

        if doc_type == "docx":
            result = parser.parse_docx(file_path)
        elif doc_type == "xlsx":
            result = parser.parse_xlsx(file_path)
        elif doc_type == "pptx":
            result = parser.parse_pptx(file_path)
        else:
            raise ValueError(f"不支持的类型: {doc_type}")

        return {
            "text": result.text,
            "markdown": result.markdown,
            "metadata": result.metadata,
            "tables": [self._table_to_dict(t) for t in result.tables],
            "sections": [self._section_to_dict(s) for s in result.sections],
            "images": result.images,
            "page_count": result.page_count,
        }

    def _extract_with_unstructured(
        self,
        file_path: Path,
        doc_type: str
    ) -> str:
        """使用 Unstructured 提取文本

        Args:
            file_path: 文件路径
            doc_type: 文档类型

        Returns:
            str: Markdown 格式的文本
        """
        from src.core.unstructured_parser import UnstructuredOfficeParser

        parser = UnstructuredOfficeParser()

        if doc_type == "docx":
            result = parser.parse_docx(file_path)
        elif doc_type == "xlsx":
            result = parser.parse_xlsx(file_path)
        elif doc_type == "pptx":
            result = parser.parse_pptx(file_path)
        else:
            raise ValueError(f"不支持的类型: {doc_type}")

        logger.info(
            f"Unstructured 解析完成: {file_path.name} - "
            f"{len(result.tables)} 表格, {len(result.sections)} 章节"
        )

        # 返回 Markdown 格式，保留结构
        return result.markdown

    def _extract_with_fallback(self, file_path: Path, doc_type: str) -> str:
        """使用原生解析器提取（作为回退）"""
        if doc_type == "pdf":
            return self._extract_pdf(file_path)
        elif doc_type == "docx":
            return self._extract_docx(file_path)
        elif doc_type == "xlsx":
            return self._extract_xlsx(file_path)
        elif doc_type == "pptx":
            return self._extract_pptx(file_path)
        elif doc_type == "image":
            return self._extract_image(file_path)
        elif doc_type == "md":
            return self._extract_md(file_path)
        elif doc_type == "txt":
            return self._extract_txt(file_path)
        elif doc_type == "code":
            return self._extract_code(file_path)
        else:
            return self._extract_txt(file_path)

    def _table_to_dict(self, table) -> Dict:
        """将 ParsedTable 转换为字典"""
        return {
            "caption": table.caption,
            "headers": table.headers,
            "rows": table.rows,
            "html": table.html,
        }

    def _section_to_dict(self, section) -> Dict:
        """将 ParsedSection 转换为字典"""
        return {
            "title": section.title,
            "level": section.level,
            "content": section.content,
            "start_page": section.start_page,
        }

    # ========== PDF 解析 ==========

    def _extract_pdf(self, file_path: Path) -> str:
        """提取 PDF 文本 - 优先使用 MinerU"""
        if self.mineru_available:
            return self._extract_pdf_with_mineru(file_path)
        else:
            return self._extract_pdf_with_pypdf(file_path)

    def _extract_pdf_with_mineru(self, file_path: Path) -> str:
        """使用 MinerU 提取 PDF（通过 Python 3.11 子进程）"""
        import subprocess
        import json

        try:
            script_path = Path(__file__).parent.parent.parent / "scripts" / "mineru.sh"
            result = subprocess.run(
                [str(script_path), str(file_path)],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode != 0:
                raise RuntimeError(f"MinerU 执行失败: {result.stderr}")

            output = result.stdout.strip()
            lines = output.split('\n')
            for line in reversed(lines):
                line = line.strip()
                if line and line.startswith('{'):
                    try:
                        data = json.loads(line)
                        if data.get("success"):
                            return data.get("text", "")
                        else:
                            raise RuntimeError(f"MinerU 处理失败: {data.get('error')}")
                    except json.JSONDecodeError:
                        continue

            return output

        except subprocess.TimeoutExpired:
            raise ValueError("MinerU 处理超时（超过2分钟）")
        except Exception as e:
            raise ValueError(f"MinerU 调用失败: {e}")

    def _extract_pdf_with_pypdf(self, file_path: Path) -> str:
        """使用 pypdf 提取 PDF"""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            text_parts = []

            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)

            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"PDF 解析失败: {e}")

    # ========== Word 解析（原生回退） ==========

    def _extract_docx(self, file_path: Path) -> str:
        """提取 Word 文档（原生解析器）"""
        try:
            from docx import Document

            doc = Document(str(file_path))
            text_parts = []

            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)

            # 尝试提取表格
            for i, table in enumerate(doc.tables, 1):
                text_parts.append(f"\n[表格 {i}]")
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    text_parts.append(row_text)

            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Word 文档解析失败: {e}")

    # ========== Excel 解析（原生回退） ==========

    def _extract_xlsx(self, file_path: Path) -> str:
        """提取 Excel 文本（原生解析器）"""
        try:
            import openpyxl

            wb = openpyxl.load_workbook(file_path, data_only=True)
            text_parts = []

            for sheet_name in wb.sheetnames:
                sheet = wb[sheet_name]
                text_parts.append(f"# 工作表: {sheet_name}")
                text_parts.append("")

                for row in sheet.iter_rows(values_only=True):
                    row_text = " | ".join(
                        str(cell) for cell in row if cell is not None
                    )
                    if row_text.strip():
                        text_parts.append(row_text)

                text_parts.append("")

            return "\n".join(text_parts)
        except ImportError:
            raise ValueError("openpyxl 未安装，无法处理 Excel 文件")
        except Exception as e:
            raise ValueError(f"Excel 解析失败: {e}")

    # ========== PowerPoint 解析（原生回退） ==========

    def _extract_pptx(self, file_path: Path) -> str:
        """提取 PowerPoint 文本（原生解析器）"""
        try:
            from pptx import Presentation

            prs = Presentation(file_path)
            text_parts = []

            for slide_num, slide in enumerate(prs.slides, 1):
                text_parts.append(f"# 幻灯片 {slide_num}")
                text_parts.append("")

                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text_parts.append(shape.text.strip())

                text_parts.append("")

            return "\n".join(text_parts)
        except ImportError:
            raise ValueError("python-pptx 未安装，无法处理 PowerPoint 文件")
        except Exception as e:
            raise ValueError(f"PowerPoint 解析失败: {e}")

    # ========== 其他格式解析 ==========

    def _extract_md(self, file_path: Path) -> str:
        """提取 Markdown"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Markdown 读取失败: {e}")

    def _extract_txt(self, file_path: Path) -> str:
        """提取纯文本"""
        try:
            encodings = ["utf-8", "gbk", "gb2312", "latin-1"]

            for encoding in encodings:
                try:
                    with open(file_path, "r", encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue

            raise ValueError("无法解码文件")
        except Exception as e:
            raise ValueError(f"文本读取失败: {e}")

    def _extract_image(self, file_path: Path) -> str:
        """提取图片文本 (OCR)"""
        try:
            from PIL import Image
            import pytesseract

            image = Image.open(file_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')

            if not text.strip():
                return f"[图片: {file_path.name}]\n\n[未识别到文本内容]"

            return f"[图片: {file_path.name}]\n\n{text}"

        except ImportError:
            raise ValueError("pytesseract 或 Pillow 未安装，无法处理图片")
        except Exception as e:
            raise ValueError(f"图片 OCR 失败: {e}")

    def _extract_code(self, file_path: Path) -> str:
        """提取代码文件 - 只提取注释
        
        代码本身不入 RAG 索引，避免分块破坏语法结构。
        只提取注释部分用于语义搜索。
        
        Args:
            file_path: 文件路径
            
        Returns:
            提取的注释内容（格式化为 Markdown）
            
        Raises:
            ValueError: 文件无注释或提取失败时抛出（不入 RAG 索引）
        """
        # 优先使用 CommentExtractor
        if self.comment_extractor:
            try:
                comments = self.comment_extractor.extract(file_path)
                if comments:
                    logger.debug(f"提取代码注释: {file_path.name} - {len(comments)} 字符")
                    return comments
                else:
                    # 没有注释，跳过索引（抛出异常让调用方处理）
                    raise ValueError(f"代码文件无注释内容，跳过索引: {file_path.name}")
            except ValueError as e:
                # 如果是"无注释"的情况，重新抛出让调用方跳过
                if "无注释" in str(e):
                    raise
                # 其他错误也跳过索引
                logger.warning(f"注释提取失败: {e}")
                raise ValueError(f"注释提取失败，跳过索引: {file_path.name}")
        
        # CommentExtractor 不可用，跳过代码文件
        raise ValueError(f"CommentExtractor 不可用，跳过代码文件: {file_path.name}")
