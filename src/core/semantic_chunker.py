"""启发式语义分块器 - 保证语义完整性 + 长度保护"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.rag_api.config import get_settings

settings = get_settings()


class BoundaryType(Enum):
    """边界类型枚举"""
    PARAGRAPH = 1      # 段落边界（\n\n）- 最强语义边界
    SECTION = 2        # 章节边界（标题）
    SENTENCE = 3       # 句子边界（。！？）
    LINE = 4           # 行边界（\n）
    LIST_ITEM = 5      # 列表项边界
    TABLE_CELL = 6     # 表格单元格边界
    WORD = 7           # 词边界（空格）
    CHARACTER = 8      # 字符边界（最后手段）


@dataclass
class SemanticBoundary:
    """语义边界信息"""
    position: int           # 边界位置（字符索引）
    boundary_type: BoundaryType
    separator: str          # 分隔符内容
    strength: float         # 边界强度（0-1，用于优先级排序）


class SemanticChunker:
    """启发式语义分块器
    
    设计原则：
    1. 语义完整性优先 - 尽量在语义边界切分
    2. 长度保护兜底 - 确保不会产生超长 chunk
    3. 多级边界查找 - 从强边界到弱边界逐级尝试
    
    架构：
    ├─ 第1层：语义边界识别（启发式规则）
    ├─ 第2层：语义分组（相邻语义单元合并）
    ├─ 第3层：长度保护（超长 chunk 在句子边界二次切分）
    └─ 第4层：重叠保护（上下文延续）
    """
    
    # 边界规则：从强到弱
    BOUNDARY_RULES: List[Tuple[BoundaryType, str, float]] = [
        # Markdown 标题（最强语义边界）
        (BoundaryType.SECTION, r'\n#{1,6}\s+', 1.0),
        # 段落边界（双换行）
        (BoundaryType.PARAGRAPH, r'\n\n+', 0.95),
        # 数字列表
        (BoundaryType.LIST_ITEM, r'\n\d+\.\s+', 0.85),
        # 无序列表
        (BoundaryType.LIST_ITEM, r'\n[•\-\*]\s+', 0.85),
        # 中文句末（。！？）
        (BoundaryType.SENTENCE, r'[。\!！\?？]\s*', 0.75),
        # 英文句末
        (BoundaryType.SENTENCE, r'[.!?:]\s+', 0.70),
        # 中文分号/逗号（较弱边界）
        (BoundaryType.SENTENCE, r'[；，;]\s*', 0.60),
        # 单换行（行边界）
        (BoundaryType.LINE, r'\n', 0.50),
        # 表格分隔符（制表符、竖线）
        (BoundaryType.TABLE_CELL, r'[\t\|]', 0.40),
        # 空格（词边界）
        (BoundaryType.WORD, r'\s+', 0.20),
    ]
    
    def __init__(
        self,
        target_chunk_size: int = None,
        max_chunk_size: int = None,
        min_chunk_size: int = None,
        chunk_overlap: int = None,
        boundary_strength_threshold: float = 0.5,
    ):
        """
        Args:
            target_chunk_size: 目标分块大小（默认 1000）
            max_chunk_size: 硬性上限（默认 8000，防止 Ollama 超限）
            min_chunk_size: 最小分块大小（默认 300，防止过碎）
            chunk_overlap: 重叠大小（默认 100）
            boundary_strength_threshold: 边界强度阈值（默认 0.5）
        """
        self.target_chunk_size = target_chunk_size or settings.CHUNK_SIZE
        self.max_chunk_size = max_chunk_size or 8000  # 硬性上限，防止 Ollama 500
        self.min_chunk_size = min_chunk_size or 300
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.boundary_strength_threshold = boundary_strength_threshold
        
    def chunk_text(self, text: str) -> List[str]:
        """对文本进行启发式语义分块
        
        流程：
        1. 预处理文本
        2. 识别所有语义边界
        3. 按边界强度分组
        4. 应用长度保护
        5. 添加重叠保护
        """
        if not text or not text.strip():
            return []
        
        # 预处理
        text = self._preprocess_text(text)
        
        # 单短文本直接返回
        if len(text) <= self.min_chunk_size:
            return [text]
        
        # Step 1: 识别所有语义边界
        boundaries = self._identify_boundaries(text)
        
        # Step 2: 基于语义边界分组
        semantic_chunks = self._group_by_semantic_boundaries(text, boundaries)
        
        # Step 3: 应用长度保护（超长 chunk 在句子边界二次切分）
        protected_chunks = self._apply_length_protection(semantic_chunks)
        
        # Step 4: 合并过小的 chunks
        merged_chunks = self._merge_small_chunks(protected_chunks)
        
        # Step 5: 添加重叠保护
        final_chunks = self._add_overlap(merged_chunks)
        
        return final_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """预处理文本
        
        处理：
        - 统一换行符
        - 去除控制字符
        - 保留必要的空白结构
        """
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 去除控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
        
        # 去除行尾空白
        text = re.sub(r'[ \t]+\n', '\n', text)
        
        # 保留段落结构（双换行），但去除多余空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _identify_boundaries(self, text: str) -> List[SemanticBoundary]:
        """识别所有语义边界
        
        Returns:
            按位置排序的边界列表
        """
        boundaries = []
        
        for boundary_type, pattern, strength in self.BOUNDARY_RULES:
            # 只查找强度达到阈值的边界
            if strength < self.boundary_strength_threshold:
                continue
            
            for match in re.finditer(pattern, text):
                # 避免边界重叠（同一位置只保留最强边界）
                pos = match.start()
                existing = [b for b in boundaries if abs(b.position - pos) < 3]
                
                if existing:
                    # 如果新边界更强，替换旧的
                    if strength > existing[0].strength:
                        boundaries.remove(existing[0])
                        boundaries.append(SemanticBoundary(
                            position=pos,
                            boundary_type=boundary_type,
                            separator=match.group(),
                            strength=strength
                        ))
                else:
                    boundaries.append(SemanticBoundary(
                        position=pos,
                        boundary_type=boundary_type,
                        separator=match.group(),
                        strength=strength
                    ))
        
        # 按位置排序
        boundaries.sort(key=lambda b: b.position)
        
        return boundaries
    
    def _group_by_semantic_boundaries(
        self, 
        text: str, 
        boundaries: List[SemanticBoundary]
    ) -> List[str]:
        """基于语义边界分组
        
        策略：
        1. 优先使用强边界（段落、标题）
        2. 当 chunk 达到目标大小时，尝试找到合适的边界切分
        3. 避免在词边界切分（除非必要）
        """
        if not boundaries:
            # 无明显边界，按目标大小切分
            return self._fallback_split(text)
        
        chunks = []
        current_start = 0
        
        # 转换为位置列表，便于查找
        boundary_positions = [(b.position, b.separator, b.strength) for b in boundaries]
        
        while current_start < len(text):
            # 计算理想结束位置
            ideal_end = current_start + self.target_chunk_size
            
            if ideal_end >= len(text):
                # 最后一段，直接加入
                chunk = text[current_start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # 在理想范围内寻找最佳边界
            best_boundary = self._find_best_boundary(
                text, 
                current_start, 
                ideal_end, 
                boundary_positions
            )
            
            if best_boundary:
                # 在边界位置切分
                end_pos = best_boundary[0] + len(best_boundary[1])
                chunk = text[current_start:end_pos].strip()
                
                if chunk:
                    chunks.append(chunk)
                current_start = end_pos
            else:
                # 未找到合适边界，强制在目标位置切分
                # 但尽量在句子或行边界
                chunk = self._safe_split_at_position(text, current_start, ideal_end)
                if chunk:
                    chunks.append(chunk)
                current_start = current_start + len(chunk)
        
        return chunks
    
    def _find_best_boundary(
        self,
        text: str,
        start: int,
        ideal_end: int,
        boundary_positions: List[Tuple[int, str, float]]
    ) -> Optional[Tuple[int, str, float]]:
        """在范围内寻找最佳切分边界
        
        Args:
            text: 文本
            start: 当前起始位置
            ideal_end: 理想结束位置
            boundary_positions: 边界位置列表
            
        Returns:
            最佳边界 (position, separator, strength) 或 None
        """
        # 定义搜索范围
        # 下限：至少达到最小 chunk 大小
        min_end = start + self.min_chunk_size
        # 上限：不超过最大 chunk 大小
        max_end = min(start + self.max_chunk_size, len(text))
        
        # 在范围内查找边界
        candidates = []
        for pos, sep, strength in boundary_positions:
            # 边界必须在范围内
            if pos < min_end:
                continue
            if pos > max_end:
                break
            
            # 计算边界的"适配度"
            # 1. 强度越高越好
            # 2. 距离理想位置越近越好
            distance_from_ideal = abs(pos - ideal_end)
            distance_penalty = distance_from_ideal / self.target_chunk_size
            
            # 综合评分：强度 * (1 - 距离惩罚)
            fitness = strength * (1 - min(distance_penalty, 0.5))
            
            candidates.append((pos, sep, strength, fitness))
        
        if not candidates:
            return None
        
        # 选择评分最高的边界
        candidates.sort(key=lambda c: c[3], reverse=True)
        best = candidates[0]
        
        return (best[0], best[1], best[2])
    
    def _safe_split_at_position(self, text: str, start: int, target_end: int) -> str:
        """安全切分：在目标位置附近寻找安全的切分点
        
        策略：
        1. 尝试在目标位置附近找到句子边界
        2. 尝试找到行边界
        3. 最后才在词边界切分
        """
        # 搜索范围：目标位置前后 20%
        search_range = int(self.target_chunk_size * 0.2)
        search_start = max(start + self.min_chunk_size, target_end - search_range)
        search_end = min(target_end + search_range, start + self.max_chunk_size, len(text))
        
        search_text = text[search_start:search_end]
        
        # 优先级：句子 > 行 > 词
        for pattern in [r'[。\!！\?？]', r'\n', r'\s']:
            matches = list(re.finditer(pattern, search_text))
            if matches:
                # 选择最接近目标位置的匹配
                best_match = min(
                    matches,
                    key=lambda m: abs(search_start + m.end() - target_end)
                )
                end_pos = search_start + best_match.end()
                return text[start:end_pos].strip()
        
        # 无法找到安全边界，强制切分
        return text[start:search_end].strip()
    
    def _apply_length_protection(self, chunks: List[str]) -> List[str]:
        """长度保护：确保所有 chunk 不超过硬性上限
        
        对超长 chunk 在句子边界二次切分
        """
        protected = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                protected.append(chunk)
            else:
                # 超长 chunk 需要二次切分
                sub_chunks = self._split_long_chunk(chunk)
                protected.extend(sub_chunks)
        
        return protected
    
    def _split_long_chunk(self, chunk: str) -> List[str]:
        """切分超长 chunk
        
        策略：在句子边界切分，确保每个子 chunk <= max_chunk_size
        """
        if len(chunk) <= self.max_chunk_size:
            return [chunk]
        
        # 识别句子边界
        sentence_boundaries = []
        for pattern in [r'[。\!！\?？]\s*', r'\n\n', r'\n']:
            for match in re.finditer(pattern, chunk):
                sentence_boundaries.append(match.end())
        
        sentence_boundaries.sort()
        
        if not sentence_boundaries:
            # 无句子边界，强制按字符切分（最后手段）
            return self._force_split_by_size(chunk)
        
        # 按句子边界切分
        sub_chunks = []
        current_start = 0
        
        for boundary in sentence_boundaries:
            if boundary - current_start >= self.max_chunk_size * 0.8:
                # 当前片段接近上限，可以切分
                if boundary - current_start <= self.max_chunk_size:
                    sub_chunks.append(chunk[current_start:boundary].strip())
                    current_start = boundary
                else:
                    # 即使在句子边界，片段仍然超长
                    # 需要进一步切分
                    oversized = chunk[current_start:boundary]
                    sub_chunks.extend(self._force_split_by_size(oversized))
                    current_start = boundary
        
        # 处理剩余部分
        if current_start < len(chunk):
            remaining = chunk[current_start:].strip()
            if remaining:
                if len(remaining) <= self.max_chunk_size:
                    sub_chunks.append(remaining)
                else:
                    sub_chunks.extend(self._force_split_by_size(remaining))
        
        return [c for c in sub_chunks if c]
    
    def _force_split_by_size(self, text: str) -> List[str]:
        """强制按大小切分（最后手段）
        
        确保每个 chunk <= max_chunk_size
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """合并过小的 chunks
        
        策略：
        - 相邻的小 chunks 合并，直到达到最小大小
        - 合并后的 chunk 不能超过目标大小
        """
        if not chunks:
            return chunks
        
        merged = []
        current = chunks[0]
        
        for chunk in chunks[1:]:
            # 判断是否需要合并
            should_merge = (
                len(current) < self.min_chunk_size or
                len(chunk) < self.min_chunk_size
            )
            
            can_merge = len(current) + len(chunk) + 2 <= self.target_chunk_size
            
            if should_merge and can_merge:
                # 合并
                current = current + "\n\n" + chunk
            else:
                merged.append(current)
                current = chunk
        
        merged.append(current)
        
        return merged
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """添加重叠保护
        
        注意：重叠后的 chunk 不能超过 max_chunk_size
        """
        if len(chunks) <= 1:
            return chunks
        
        overlapped = []
        
        for i, chunk in enumerate(chunks):
            # 添加前一个 chunk 的结尾作为重叠
            if i > 0 and self.chunk_overlap > 0:
                prev_chunk = chunks[i - 1]
                overlap_text = self._get_overlap_text(prev_chunk, self.chunk_overlap)
                
                # 确保添加重叠后不超过上限
                if len(overlap_text) + len(chunk) <= self.max_chunk_size:
                    chunk = overlap_text + chunk
            
            overlapped.append(chunk)
        
        return overlapped
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """获取重叠文本
        
        从文本结尾提取 overlap_size 的内容
        尽量在句子边界截取
        """
        if overlap_size <= 0 or len(text) <= overlap_size:
            return ""
        
        # 从结尾向前提取
        start = len(text) - overlap_size
        
        # 尝试找到句子边界作为重叠起始点
        search_text = text[start:start + 50]  # 搜索前 50 字符
        
        for pattern in [r'[。\!！\?？]\s*', r'\n', r'\s']:
            match = re.search(pattern, search_text)
            if match:
                # 找到边界，从边界后开始重叠
                boundary_pos = start + match.end()
                return text[boundary_pos:].strip()
        
        # 未找到边界，直接从 start 开始
        return text[start:].strip()
    
    def _fallback_split(self, text: str) -> List[str]:
        """兜底切分方法
        
        当没有识别到任何语义边界时使用
        """
        # 尝试按段落切分
        paragraphs = re.split(r'\n\n+', text)
        
        if len(paragraphs) > 1:
            # 按段落分组
            chunks = []
            current = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current) + len(para) + 2 <= self.target_chunk_size:
                    current = current + "\n\n" + para if current else para
                else:
                    if current:
                        chunks.append(current)
                    current = para
                    
                    # 如果单个段落就超长，需要进一步切分
                    if len(current) > self.max_chunk_size:
                        sub_chunks = self._split_long_chunk(current)
                        chunks.extend(sub_chunks[:-1])
                        current = sub_chunks[-1] if sub_chunks else ""
            
            if current:
                chunks.append(current)
            
            return chunks
        
        # 没有段落边界，按句子切分
        sentences = re.split(r'[。\!！\?？\n]', text)
        
        if len(sentences) > 1:
            chunks = []
            current = ""
            
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                
                if len(current) + len(sent) + 1 <= self.target_chunk_size:
                    current = current + sent if current else sent
                else:
                    if current:
                        chunks.append(current)
                    current = sent
            
            if current:
                chunks.append(current)
            
            return chunks
        
        # 最后手段：强制按大小切分
        return self._force_split_by_size(text)
    
    def chunk_text_with_metadata(self, text: str, file_path: str = None) -> List[Dict[str, Any]]:
        """分块并返回带元数据的结果
        
        Args:
            text: 待分块文本
            file_path: 文件路径
            
        Returns:
            分块列表，每个分块包含 content, start_char, end_char, start_line, end_line, metadata
        """
        if not text:
            return []
        
        # 预计算换行符位置，用于精确行号计算
        newline_positions = self._get_newline_positions(text)
        
        chunks = self.chunk_text(text)
        
        # 计算每个 chunk 的字符位置和行号
        result = []
        current_pos = 0
        
        for chunk in chunks:
            # 在原文中查找 chunk 的位置（改进的定位逻辑）
            start = self._find_chunk_position(text, chunk, current_pos)
            end = start + len(chunk)
            
            # 计算精确行号
            start_line = self._char_to_line(start, newline_positions)
            end_line = self._char_to_line(end - 1, newline_positions)  # end-1 因为 end 是结束位置的下一个字符
            
            result.append({
                "content": chunk,
                "start_char": start,
                "end_char": end,
                "start_line": start_line,
                "end_line": end_line,
                "length": len(chunk),
                "metadata": {
                    "file_path": file_path,
                    "chunk_index": len(result),
                }
            })
            
            current_pos = end
        
        return result
    
    def _get_newline_positions(self, text: str) -> List[int]:
        """获取所有换行符的位置
        
        Returns:
            换行符位置列表（按顺序排列）
        """
        positions = []
        for i, char in enumerate(text):
            if char == '\n':
                positions.append(i)
        return positions
    
    def _char_to_line(self, char_pos: int, newline_positions: List[int]) -> int:
        """将字符位置转换为行号
        
        Args:
            char_pos: 字符位置（0-based）
            newline_positions: 换行符位置列表
            
        Returns:
            行号（1-based）
        """
        # 行号 = 换行符数量 + 1（在当前位置之前）
        line_num = 1
        for pos in newline_positions:
            if pos < char_pos:
                line_num += 1
            else:
                break
        return line_num
    
    def _find_chunk_position(self, text: str, chunk: str, start_pos: int) -> int:
        """在原文中查找 chunk 的精确位置
        
        Args:
            text: 原文
            chunk: 分块内容
            start_pos: 搜索起始位置
            
        Returns:
            chunk 在原文中的起始位置
        """
        # 策略1：从上次位置开始精确查找
        search_pos = start_pos
        
        # 使用 chunk 的前 30 字符作为定位锚点（减少重复内容误匹配）
        anchor_len = min(30, len(chunk))
        anchor = chunk[:anchor_len]
        
        # 多次尝试查找，逐步扩大搜索范围
        max_attempts = 3
        for attempt in range(max_attempts):
            pos = text.find(anchor, search_pos)
            
            if pos != -1:
                # 验证：检查从该位置开始的内容是否匹配整个 chunk
                candidate = text[pos:pos + len(chunk)]
                if candidate == chunk:
                    return pos
                
                # 部分匹配，可能是重复内容，继续搜索
                search_pos = pos + 1
            
            # 扩大搜索范围
            search_pos = start_pos
        
        # 策略2：如果精确查找失败，使用更长的锚点（前 100 字符）
        if len(chunk) >= 100:
            longer_anchor = chunk[:100]
            pos = text.find(longer_anchor, start_pos)
            if pos != -1:
                return pos
        
        # 策略3：返回上次位置作为近似值（最后手段）
        return start_pos


def get_semantic_chunker() -> SemanticChunker:
    """获取语义分块器实例"""
    return SemanticChunker()