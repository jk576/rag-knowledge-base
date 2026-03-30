"""代码注释提取器

从代码文件中提取注释部分，用于 RAG 索引。
只提取注释，不提取代码本身，避免分块破坏语法结构。
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class CommentExtractor:
    """代码注释提取器
    
    支持多种编程语言的注释提取：
    - Python: 井号单行注释, 三引号多行docstring
    - JavaScript/TypeScript: 双斜杠单行, 斜杠星号多行
    - Java/C/C++: 双斜杠单行, 斜杠星号多行
    - Go: 双斜杠单行
    - Rust: 双斜杠单行, 斜杠星号多行
    - Shell: 井号单行
    """
    
    def __init__(self):
        # 各语言的注释模式（在 __init__ 中定义以避免 raw string 问题）
        self.COMMENT_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
            'python': [
                (r'#.*$', 'single'),                     # 单行注释
                (r'"{3}[\s\S]*?"{3}', 'multi'),          # Python docstring (双引号)
                (r"'{3}[\s\S]*?'{3}", 'multi'),          # Python docstring (单引号)
            ],
            'javascript': [
                (r'//.*$', 'single'),                    # 单行注释
                (r'/[*][\s\S]*?[*]/', 'multi'),          # 多行注释
            ],
            'typescript': [
                (r'//.*$', 'single'),
                (r'/[*][\s\S]*?[*]/', 'multi'),
            ],
            'java': [
                (r'//.*$', 'single'),
                (r'/[*][\s\S]*?[*]/', 'multi'),
            ],
            'c': [
                (r'//.*$', 'single'),
                (r'/[*][\s\S]*?[*]/', 'multi'),
            ],
            'cpp': [
                (r'//.*$', 'single'),
                (r'/[*][\s\S]*?[*]/', 'multi'),
            ],
            'go': [
                (r'//.*$', 'single'),
            ],
            'rust': [
                (r'//.*$', 'single'),
                (r'/[*][\s\S]*?[*]/', 'multi'),
            ],
            'shell': [
                (r'#.*$', 'single'),
            ],
            'ruby': [
                (r'#.*$', 'single'),
            ],
            'php': [
                (r'//.*$', 'single'),
                (r'#.*$', 'single'),
                (r'/[*][\s\S]*?[*]/', 'multi'),
            ],
        }
        
        # 文件扩展名到语言的映射
        self.EXT_TO_LANG: Dict[str, str] = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.c': 'c',
            '.h': 'c',
            '.cpp': 'cpp',
            '.hpp': 'cpp',
            '.cc': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            '.rb': 'ruby',
            '.php': 'php',
        }
        
        # 支持代码注释提取的文件类型
        self.CODE_EXTENSIONS = set(self.EXT_TO_LANG.keys())
        
        # 预编译正则表达式以提高性能
        # 单行注释不使用 DOTALL，多行注释使用 DOTALL
        self._compiled_patterns: Dict[str, List[Tuple[re.Pattern, str]]] = {}
        for lang, patterns in self.COMMENT_PATTERNS.items():
            compiled = []
            for pattern, type_ in patterns:
                if type_ == 'multi':
                    # 多行注释需要 DOTALL 让 . 匹配换行符
                    compiled.append((re.compile(pattern, re.MULTILINE | re.DOTALL), type_))
                else:
                    # 单行注释不需要 DOTALL
                    compiled.append((re.compile(pattern, re.MULTILINE), type_))
            self._compiled_patterns[lang] = compiled
    
    def get_language(self, file_path: Union[str, Path]) -> Optional[str]:
        """根据文件扩展名获取语言类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            语言类型，如果不支持则返回 None
        """
        ext = Path(file_path).suffix.lower()
        return self.EXT_TO_LANG.get(ext)
    
    def is_code_file(self, file_path: Union[str, Path]) -> bool:
        """检查是否是支持注释提取的代码文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否是代码文件
        """
        ext = Path(file_path).suffix.lower()
        return ext in self.CODE_EXTENSIONS
    
    def extract(self, file_path: Union[str, Path]) -> str:
        """提取代码文件中的所有注释
        
        Args:
            file_path: 文件路径
            
        Returns:
            提取的注释内容，格式化为 Markdown
            
        Raises:
            ValueError: 文件类型不支持或读取失败
        """
        file_path = Path(file_path)
        language = self.get_language(file_path)
        
        if language not in self._compiled_patterns:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"文件读取失败: {e}")
        
        # 提取注释
        comments = self._extract_comments(content, language)
        
        if not comments:
            logger.debug(f"文件 {file_path} 中没有找到注释")
            return ""
        
        # 格式化输出
        return self._format_comments(file_path, comments)
    
    def extract_from_content(self, content: str, language: str) -> str:
        """从内容中提取注释（不读取文件）
        
        Args:
            content: 代码内容
            language: 语言类型
            
        Returns:
            提取的注释内容
        """
        if language not in self._compiled_patterns:
            raise ValueError(f"不支持的语言: {language}")
        
        comments = self._extract_comments(content, language)
        
        if not comments:
            return ""
        
        return "\n\n".join(comments)
    
    def _extract_comments(self, content: str, language: str) -> List[str]:
        """提取所有注释
        
        Args:
            content: 代码内容
            language: 语言类型
            
        Returns:
            注释列表
        """
        comments: List[str] = []
        
        for pattern, type_ in self._compiled_patterns[language]:
            matches = pattern.findall(content)
            
            for match in matches:
                # 清理注释内容
                cleaned = self._clean_comment(match, type_)
                if cleaned.strip():
                    comments.append(cleaned)
        
        return comments
    
    def _clean_comment(self, comment: str, type_: str) -> str:
        """清理注释内容
        
        - 移除注释符号
        - 移除多余的空白
        - 保留有意义的注释内容
        
        Args:
            comment: 原始注释
            type_: 注释类型（single/multi）
            
        Returns:
            清理后的注释
        """
        if type_ == 'single':
            # 移除 # 或 // 前缀
            if comment.startswith('#'):
                cleaned = comment[1:].strip()
            elif comment.startswith('//'):
                cleaned = comment[2:].strip()
            else:
                cleaned = comment.strip()
            
            # 过滤无意义的注释
            if self._is_meaningless(cleaned):
                return ""
            
            return cleaned
        
        elif type_ == 'multi':
            # 移除 Python docstring 或 C-style comment 包围
            cleaned = comment.strip()
            
            # Python docstring - 使用字符拼接避免三引号语法问题
            triple_double = '"' + '"' + '"'
            triple_single = "'" + "'" + "'"
            
            if cleaned.startswith(triple_double) and cleaned.endswith(triple_double):
                cleaned = cleaned[3:-3].strip()
            elif cleaned.startswith(triple_single) and cleaned.endswith(triple_single):
                cleaned = cleaned[3:-3].strip()
            
            # C-style comment
            elif cleaned.startswith('/*') and cleaned.endswith('*/'):
                cleaned = cleaned[2:-2].strip()
            
            # 处理多行注释中的每行
            lines = []
            for line in cleaned.split('\n'):
                line = line.strip()
                # 移除行首的 *（常见于多行注释格式）
                if line.startswith('*'):
                    line = line[1:].strip()
                if line and not self._is_meaningless(line):
                    lines.append(line)
            
            return '\n'.join(lines) if lines else ""
        
        return comment.strip()
    
    def _is_meaningless(self, comment: str) -> bool:
        """判断注释是否无意义
        
        过滤以下类型的注释：
        - 空注释
        - 纯分隔符（如 # ----）
        - 简单的 TODO/FIXME 标记（没有具体说明）
        - 自动生成的注释（如 IDE 生成的）
        
        Args:
            comment: 注释内容
            
        Returns:
            是否无意义
        """
        if not comment.strip():
            return True
        
        # 纯分隔符
        if re.match(r'^[-=_\s*]+$', comment):
            return True
        
        # 简单的 TODO/FIXME（没有具体说明）
        if re.match(r'^TODO[:\s]*$', comment, re.IGNORECASE):
            return True
        if re.match(r'^FIXME[:\s]*$', comment, re.IGNORECASE):
            return True
        
        # 短注释（少于 3 个字符）
        if len(comment.strip()) < 3:
            return True
        
        return False
    
    def _format_comments(self, file_path: Path, comments: List[str]) -> str:
        """格式化注释输出
        
        Args:
            file_path: 文件路径
            comments: 注释列表
            
        Returns:
            格式化的 Markdown 文本
        """
        parts = [
            f"# File: {file_path.name}",
            f"# Path: {file_path}",
            "",
            "## Comments",
            "",
        ]
        
        for i, comment in enumerate(comments, 1):
            # 如果是多行注释，保持格式
            if '\n' in comment:
                parts.append(f"### Comment {i}")
                parts.append("")
                parts.append(comment)
                parts.append("")
            else:
                parts.append(f"{i}. {comment}")
        
        return '\n'.join(parts)


def extract_code_comments(file_path: Union[str, Path]) -> str:
    """便捷函数：提取代码注释
    
    Args:
        file_path: 文件路径
        
    Returns:
        提取的注释内容
    """
    extractor = CommentExtractor()
    return extractor.extract(file_path)