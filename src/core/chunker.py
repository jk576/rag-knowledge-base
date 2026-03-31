"""文本分块器 - 统一入口"""

import re
from typing import List, Dict, Any, Union
from pathlib import Path

from src.rag_api.config import get_settings
from src.core.semantic_chunker import SemanticChunker, get_semantic_chunker

settings = get_settings()


class ChunkWithMetadata:
    """带元数据的分块"""
    def __init__(self, content: str, start_line: int = 0, end_line: int = 0, 
                 metadata: Dict[str, Any] = None):
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata
        }


class TextChunker:
    """文本分块器 - 统一入口
    
    根据 USE_SEMANTIC_CHUNKING 配置选择分块策略：
    - True: 使用启发式语义分块（推荐）
    - False: 使用传统分块方法
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
        use_semantic: bool = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = separators or settings.CHUNK_SEPARATORS
        
        # 决定使用哪种分块策略
        self.use_semantic = use_semantic if use_semantic is not None else settings.USE_SEMANTIC_CHUNKING
        
        if self.use_semantic:
            # 使用启发式语义分块器
            self._semantic_chunker = SemanticChunker(
                target_chunk_size=self.chunk_size,
                max_chunk_size=settings.MAX_CHUNK_SIZE,
                min_chunk_size=settings.MIN_CHUNK_SIZE,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            self._semantic_chunker = None
    
    def chunk_text(self, text: str) -> List[str]:
        """对文本进行分块
        
        自动选择分块策略：
        - 启用语义分块：使用启发式语义边界 + 长度保护
        - 未启用：使用传统分块方法
        """
        if not text:
            return []
        
        if self.use_semantic and self._semantic_chunker:
            # 使用启发式语义分块
            return self._semantic_chunker.chunk_text(text)
        
        # 传统分块方法（向后兼容）
        return self._legacy_chunk_text(text)
    
    def chunk_text_with_location(self, text: str, file_path: str = None) -> List[ChunkWithMetadata]:
        """对文本进行分块，保留行号信息
        
        自动选择分块策略：
        - 启用语义分块：使用启发式语义边界 + 长度保护
        - 未启用：使用传统分块方法
        """
        if not text:
            return []
        
        # 优先使用语义分块器
        if self.use_semantic and self._semantic_chunker:
            semantic_chunks = self._semantic_chunker.chunk_text_with_metadata(text, file_path)
            # 转换为 ChunkWithMetadata 格式
            # 注意：语义分块器返回的 start_line/end_line 在顶层，不在 metadata 里
            return [
                ChunkWithMetadata(
                    content=c["content"],
                    start_line=c.get("start_line", 1),
                    end_line=c.get("end_line", 1),
                    metadata=c.get("metadata", {})
                )
                for c in semantic_chunks
            ]
        
        # 传统分块方法（向后兼容）
        lines = text.split('\n')
        total_lines = len(lines)
        
        # 如果文本较短，直接返回
        if len(text) <= self.chunk_size:
            return [ChunkWithMetadata(
                content=text.strip(),
                start_line=1,
                end_line=total_lines,
                metadata={"file_path": file_path}
            )] if text.strip() else []
        
        chunks = []
        current_chunk_lines = []
        current_size = 0
        chunk_start_line = 1
        
        for line_num, line in enumerate(lines, 1):
            line_with_newline = line + '\n'
            line_size = len(line_with_newline)
            
            # 如果当前行加入后超过块大小，先保存当前块
            if current_size + line_size > self.chunk_size and current_chunk_lines:
                chunk_text = ''.join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append(ChunkWithMetadata(
                        content=chunk_text,
                        start_line=chunk_start_line,
                        end_line=line_num - 1,
                        metadata={"file_path": file_path}
                    ))
                
                # 处理重叠：保留部分行用于下一个块
                overlap_lines = self._calculate_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines + [line_with_newline]
                current_size = sum(len(l) for l in current_chunk_lines)
                chunk_start_line = line_num - len(overlap_lines)
            else:
                current_chunk_lines.append(line_with_newline)
                current_size += line_size
        
        # 保存最后一个块
        if current_chunk_lines:
            chunk_text = ''.join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append(ChunkWithMetadata(
                    content=chunk_text,
                    start_line=chunk_start_line,
                    end_line=total_lines,
                    metadata={"file_path": file_path}
                ))
        
        return chunks
    
    def _calculate_overlap_lines(self, lines: List[str]) -> List[str]:
        """计算重叠行"""
        overlap_size = 0
        overlap_lines = []
        
        for line in reversed(lines):
            if overlap_size + len(line) <= self.chunk_overlap:
                overlap_lines.insert(0, line)
                overlap_size += len(line)
            else:
                break
        
        return overlap_lines
    
    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        # 去除多余空白
        text = re.sub(r"\s+", " ", text)
        # 去除多余空行
        text = re.sub(r"\n\s*\n", "\n\n", text)
        return text.strip()
    
    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """按分隔符分块"""
        if not separator:
            return self._split_by_characters(text)
        
        parts = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # 如果当前块加上新部分超过限制
            if len(current_chunk) + len(part) + len(separator) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # 如果单个部分就超过限制，需要进一步切分
                if len(part) > self.chunk_size:
                    sub_chunks = self._split_by_characters(part)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = part
            else:
                if current_chunk:
                    current_chunk += separator + part
                else:
                    current_chunk = part
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_by_characters(self, text: str) -> List[str]:
        """按字符数量强制切分"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # 尝试在句子边界切分
            chunk = text[start:end]
            # 查找最后一个句号或换行
            for sep in ["\n\n", "。", "；", "\n", " "]:
                last_sep = chunk.rfind(sep)
                if last_sep > self.chunk_size * 0.5:  # 至少保留一半内容
                    end = start + last_sep + len(sep)
                    break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个起始位置（考虑重叠）
            next_start = end - self.chunk_overlap
            
            # 边界检查：防止死循环和内容丢失
            if next_start >= len(text):
                # 已到达末尾，退出
                break
            
            if next_start <= start:
                # 重叠过大或重叠为负，从当前位置继续
                # 确保至少前进一小步，避免死循环
                next_start = min(start + 1, len(text) - 1)
            
            start = next_start
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的块（传统方法）"""
        if not chunks:
            return chunks
        
        min_size = self.chunk_size * 0.3  # 最小30%
        max_size = settings.MAX_CHUNK_SIZE  # 确保不超过硬性上限
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            # 合并条件：当前块太小，且合并后不超过上限
            if len(current) < min_size and len(current) + len(chunk) + 2 <= max_size:
                current += "\n\n" + chunk
            else:
                merged.append(current)
                current = chunk
        
        merged.append(current)
        
        # 最终检查：确保所有 chunk 不超过硬性上限
        final_merged = []
        for chunk in merged:
            if len(chunk) > max_size:
                # 超长 chunk 需要二次切分
                sub_chunks = self._split_overlong_chunk(chunk)
                final_merged.extend(sub_chunks)
            else:
                final_merged.append(chunk)
        
        return final_merged
    
    def _split_overlong_chunk(self, chunk: str) -> List[str]:
        """切分超长 chunk（确保不超过 MAX_CHUNK_SIZE）"""
        max_size = settings.MAX_CHUNK_SIZE
        
        if len(chunk) <= max_size:
            return [chunk]
        
        # 尝试在句子边界切分
        sub_chunks = []
        start = 0
        
        while start < len(chunk):
            end = min(start + max_size, len(chunk))
            
            if end < len(chunk):
                # 尝试找到边界
                search_text = chunk[start:end]
                for sep in ["\n\n", "。", "；", "\n", " ", ""]:
                    last_sep = search_text.rfind(sep)
                    if last_sep > max_size * 0.5:
                        end = start + last_sep + len(sep)
                        break
            
            sub_chunk = chunk[start:end].strip()
            if sub_chunk:
                sub_chunks.append(sub_chunk)
            start = end
        
        return sub_chunks
    
    def _legacy_chunk_text(self, text: str) -> List[str]:
        """传统分块方法（向后兼容）"""
        # 清洗文本
        text = self._clean_text(text)
        
        # 硬性上限检查：如果文本本身就超长，需要特殊处理
        max_size = settings.MAX_CHUNK_SIZE
        if len(text) > max_size * 2:
            # 超长文本，需要先切分再处理
            return self._split_overlong_text(text)
        
        # 如果文本长度小于块大小，直接返回
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        chunks = []
        
        # 尝试按分隔符分块
        for separator in self.separators:
            chunks = self._split_by_separator(text, separator)
            if len(chunks) > 1:
                break
        
        # 如果还是没有分块，强制按字符切分
        if len(chunks) <= 1:
            chunks = self._split_by_characters(text)
        
        # 合并小片段（确保不超过上限）
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _split_overlong_text(self, text: str) -> List[str]:
        """切分超长文本（特殊处理）"""
        max_size = settings.MAX_CHUNK_SIZE
        
        # 先按段落切分
        paragraphs = text.split("\n\n")
        
        chunks = []
        current = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current) + len(para) + 2 <= max_size:
                current = current + "\n\n" + para if current else para
            else:
                if current:
                    chunks.append(current)
                
                # 如果单个段落就超长，需要进一步切分
                if len(para) > max_size:
                    sub_chunks = self._split_by_characters(para)
                    chunks.extend(sub_chunks)
                else:
                    current = para
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def chunk_markdown(self, text: str) -> List[dict]:
        """对 Markdown 按标题分块"""
        import re
        
        # 匹配标题
        header_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)
        
        matches = list(header_pattern.finditer(text))
        
        if not matches:
            return [{"header": "", "content": text, "level": 0}]
        
        chunks = []
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section = text[start:end].strip()
            header_line = match.group(1)
            level = len(header_line) - len(header_line.lstrip("#"))
            header_text = header_line.lstrip("#").strip()
            
            # 进一步切分大块
            content = section[len(header_line):].strip()
            if len(content) > self.chunk_size:
                sub_chunks = self.chunk_text(content)
                for j, sub in enumerate(sub_chunks):
                    chunks.append({
                        "header": f"{header_text} (part {j+1})" if j > 0 else header_text,
                        "content": sub,
                        "level": level,
                    })
            else:
                chunks.append({
                    "header": header_text,
                    "content": content,
                    "level": level,
                })
        
        return chunks

    def chunk_code_with_symbols(self, text: str, file_path: str = None, language: str = None) -> List[ChunkWithMetadata]:
        """对代码文件进行语义分块，识别函数/类/方法边界"""
        if not text:
            return []
        
        lines = text.split('\n')
        total_lines = len(lines)
        
        # 根据语言选择解析规则
        patterns = self._get_code_patterns(language)
        
        chunks = []
        current_chunk_lines = []
        current_chunk_metadata = {"file_path": file_path, "symbols": []}
        chunk_start_line = 1
        current_size = 0
        
        for line_num, line in enumerate(lines, 1):
            line_with_newline = line + '\n'
            line_size = len(line_with_newline)
            
            # 检测代码符号（函数、类定义等）
            symbol = self._detect_symbol(line, patterns, line_num)
            if symbol:
                # 如果当前块已经有内容，先保存
                if current_chunk_lines and current_size > self.chunk_size * 0.3:
                    chunk_text = ''.join(current_chunk_lines).strip()
                    if chunk_text:
                        chunks.append(ChunkWithMetadata(
                            content=chunk_text,
                            start_line=chunk_start_line,
                            end_line=line_num - 1,
                            metadata=current_chunk_metadata.copy()
                        ))
                    current_chunk_lines = []
                    current_size = 0
                    chunk_start_line = line_num
                    current_chunk_metadata = {"file_path": file_path, "symbols": []}
                
                current_chunk_metadata["symbols"].append(symbol)
            
            current_chunk_lines.append(line_with_newline)
            current_size += line_size
            
            # 如果超过块大小，保存当前块
            if current_size >= self.chunk_size and current_chunk_lines:
                chunk_text = ''.join(current_chunk_lines).strip()
                if chunk_text:
                    chunks.append(ChunkWithMetadata(
                        content=chunk_text,
                        start_line=chunk_start_line,
                        end_line=line_num,
                        metadata=current_chunk_metadata.copy()
                    ))
                
                # 重叠处理
                overlap_lines = self._calculate_overlap_lines(current_chunk_lines)
                current_chunk_lines = overlap_lines
                current_size = sum(len(l) for l in overlap_lines)
                chunk_start_line = line_num - len(overlap_lines) + 1
                current_chunk_metadata = {"file_path": file_path, "symbols": []}
        
        # 保存最后一个块
        if current_chunk_lines:
            chunk_text = ''.join(current_chunk_lines).strip()
            if chunk_text:
                chunks.append(ChunkWithMetadata(
                    content=chunk_text,
                    start_line=chunk_start_line,
                    end_line=total_lines,
                    metadata=current_chunk_metadata
                ))
        
        return chunks
    
    def _get_code_patterns(self, language: str = None) -> Dict[str, Any]:
        """获取代码解析模式"""
        # 通用模式
        common_patterns = {
            'function': r'^\s*(?:function|def|func)\s+\w+|^\s*(?:const|let|var)?\s*\w+\s*=\s*(?:async\s*)?\(|^\s*\w+\s*:\s*(?:async\s*)?\(',
            'class': r'^\s*(?:class|interface|struct)\s+\w+',
            'method': r'^\s*(?:public|private|protected|static)?\s*(?:async\s*)?\w+\s*\(',
        }
        
        # 语言特定模式
        lang_patterns = {
            'python': {
                'function': r'^\s*def\s+\w+',
                'class': r'^\s*class\s+\w+',
            },
            'typescript': {
                'function': r'^\s*(?:export\s+)?(?:async\s+)?function\s+\w+|^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(',
                'class': r'^\s*(?:export\s+)?class\s+\w+',
                'interface': r'^\s*(?:export\s+)?interface\s+\w+',
            },
            'javascript': {
                'function': r'^\s*(?:export\s+)?(?:async\s+)?function\s+\w+|^\s*(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?\(',
                'class': r'^\s*(?:export\s+)?class\s+\w+',
            },
        }
        
        if language and language in lang_patterns:
            return {**common_patterns, **lang_patterns[language]}
        return common_patterns
    
    def _detect_symbol(self, line: str, patterns: Dict[str, Any], line_num: int) -> Dict[str, Any]:
        """检测代码符号"""
        for symbol_type, pattern in patterns.items():
            if re.match(pattern, line):
                # 提取符号名称
                name_match = re.search(r'(?:class|interface|struct|function|def|func)\s+(\w+)', line)
                if name_match:
                    return {
                        "type": symbol_type,
                        "name": name_match.group(1),
                        "line": line_num,
                        "signature": line.strip()[:100]  # 保留前100字符作为签名
                    }
        return None
