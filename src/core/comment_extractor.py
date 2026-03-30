"""代码注释提取器（改进版）

从代码文件中提取注释部分，用于 RAG 索引。
只提取注释，不提取代码本身，避免分块破坏语法结构。

改进点：
1. Python 使用 AST 解析，准确区分注释和字符串
2. 其他语言使用 token-based 解析（尽可能准确）
3. 文件类型配置统一管理
"""

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# 文件类型配置（统一管理，避免重复定义）
# ============================================================================

DOC_EXTENSIONS: Set[str] = {
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
    ".md", ".txt", ".rst",
}

CODE_EXTENSIONS: Set[str] = {
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".java", ".go", ".rs",
    ".cpp", ".c", ".h", ".hpp", ".cc",
    ".sh", ".bash", ".zsh",
    ".rb", ".php",
}

IMAGE_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp",
}

SUPPORTED_EXTENSIONS = DOC_EXTENSIONS | CODE_EXTENSIONS | IMAGE_EXTENSIONS

# 文件大小限制（避免处理超大文件）
MAX_FILE_SIZE_MB = 10  # 代码文件最大 10MB
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def get_file_category(file_path: Union[str, Path]) -> str:
    """获取文件类别
    
    Args:
        file_path: 文件路径
        
    Returns:
        文件类别: 'doc' | 'code' | 'image' | 'unknown'
    """
    ext = Path(file_path).suffix.lower()
    
    if ext in DOC_EXTENSIONS:
        return 'doc'
    elif ext in CODE_EXTENSIONS:
        return 'code'
    elif ext in IMAGE_EXTENSIONS:
        return 'image'
    else:
        return 'unknown'


# ============================================================================
# 注释提取器
# ============================================================================

class CommentExtractor:
    """代码注释提取器（改进版）
    
    支持多种编程语言的注释提取，使用不同的策略：
    - Python: AST 解析（最准确，区分注释和字符串）
    - 其他语言: 正则 + token 解析（尽力而为）
    """
    
    # 文件扩展名到语言的映射
    EXT_TO_LANG: Dict[str, str] = {
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
    
    def __init__(self):
        # 非 Python 语言的注释模式
        self._comment_patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[str, List[Tuple[re.Pattern, str]]]:
        """初始化非 Python 语言的注释模式"""
        patterns = {}
        
        # JavaScript/TypeScript/Java/C/C++/Go/Rust/PHP
        c_style_langs = ['javascript', 'typescript', 'java', 'c', 'cpp', 'go', 'rust', 'php']
        for lang in c_style_langs:
            patterns[lang] = [
                (re.compile(r'//[^\n]*', re.MULTILINE), 'single'),
                (re.compile(r'/[*][\s\S]*?[*]/', re.MULTILINE | re.DOTALL), 'multi'),
            ]
        
        # Shell/Ruby
        shell_langs = ['shell', 'ruby']
        for lang in shell_langs:
            patterns[lang] = [
                (re.compile(r'#[^\n]*', re.MULTILINE), 'single'),
            ]
        
        return patterns
    
    def get_language(self, file_path: Union[str, Path]) -> Optional[str]:
        """根据文件扩展名获取语言类型"""
        ext = Path(file_path).suffix.lower()
        return self.EXT_TO_LANG.get(ext)
    
    def is_code_file(self, file_path: Union[str, Path]) -> bool:
        """检查是否是支持注释提取的代码文件"""
        ext = Path(file_path).suffix.lower()
        return ext in CODE_EXTENSIONS
    
    def extract(self, file_path: Union[str, Path]) -> str:
        """提取代码文件中的所有注释
        
        Args:
            file_path: 文件路径
            
        Returns:
            提取的注释内容，格式化为 Markdown
            如果没有注释则返回空字符串
            
        Raises:
            ValueError: 文件类型不支持、文件过大或读取失败
        """
        file_path = Path(file_path)
        language = self.get_language(file_path)
        
        if not language:
            raise ValueError(f"不支持的文件类型: {file_path.suffix}")
        
        # 检查文件大小
        try:
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"文件过大 ({file_size / 1024 / 1024:.1f}MB > {MAX_FILE_SIZE_MB}MB)，跳过处理"
                )
        except FileNotFoundError:
            raise ValueError(f"文件不存在: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"文件读取失败: {e}")
        
        # Python 使用 AST 解析（最准确）
        if language == 'python':
            comments = self._extract_python_comments(content, file_path)
        else:
            # 其他语言使用正则
            comments = self._extract_generic_comments(content, language)
        
        if not comments:
            logger.debug(f"文件 {file_path} 中没有找到注释")
            return ""  # 无注释返回空字符串，不返回占位信息
        
        return self._format_comments(file_path, comments)
    
    def _extract_python_comments(self, content: str, file_path: Path) -> List[str]:
        """使用 AST 提取 Python 注释（准确区分注释和字符串）
        
        Args:
            content: Python 代码内容
            file_path: 文件路径（用于错误报告）
            
        Returns:
            注释列表
        """
        comments: List[str] = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Python 语法错误: {file_path}: {e}")
            # 回退到简单正则
            return self._extract_generic_comments(content, 'python')
        
        # 提取所有注释（AST 只能获取 docstring，行注释需要 tokenize）
        # 使用 tokenize 模块
        import tokenize
        import io
        
        try:
            tokens = tokenize.generate_tokens(io.StringIO(content).readline)
            for token in tokens:
                if token.type == tokenize.COMMENT:
                    # 行注释
                    comment_text = token.string[1:].strip()  # 去掉 #
                    if comment_text and not self._is_meaningless(comment_text):
                        comments.append(comment_text)
                elif token.type == tokenize.STRING:
                    # 检查是否是 docstring
                    # docstring 是模块/类/函数的第一个语句
                    pass  # AST 已经处理了 docstring
        except tokenize.TokenError as e:
            logger.warning(f"Token 错误: {file_path}: {e}")
        
        # 提取 docstring
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                if docstring and docstring.strip():
                    cleaned = self._clean_docstring(docstring)
                    if cleaned:
                        comments.append(f"[{node.__class__.__name__} {node.name}] {cleaned}")
            
            # 模块级 docstring
            if isinstance(node, ast.Module):
                docstring = ast.get_docstring(node)
                if docstring and docstring.strip():
                    cleaned = self._clean_docstring(docstring)
                    if cleaned:
                        comments.insert(0, f"[Module] {cleaned}")
        
        return comments
    
    def _extract_generic_comments(self, content: str, language: str) -> List[str]:
        """使用正则提取注释（非 Python 语言）
        
        注意：正则方法无法区分字符串中的假注释，这是已知的局限性。
        
        Args:
            content: 代码内容
            language: 语言类型
            
        Returns:
            注释列表
        """
        if language not in self._comment_patterns:
            return []
        
        comments: List[str] = []
        
        for pattern, type_ in self._comment_patterns[language]:
            matches = pattern.findall(content)
            
            for match in matches:
                cleaned = self._clean_comment(match, type_)
                if cleaned.strip():
                    comments.append(cleaned)
        
        return comments
    
    def _clean_comment(self, comment: str, type_: str) -> str:
        """清理注释内容"""
        if type_ == 'single':
            # 移除 # 或 // 前缀
            if comment.startswith('#'):
                cleaned = comment[1:].strip()
            elif comment.startswith('//'):
                cleaned = comment[2:].strip()
            else:
                cleaned = comment.strip()
            
            if self._is_meaningless(cleaned):
                return ""
            return cleaned
        
        elif type_ == 'multi':
            # 移除 /* */ 包围
            cleaned = comment.strip()
            if cleaned.startswith('/*') and cleaned.endswith('*/'):
                cleaned = cleaned[2:-2].strip()
            
            # 处理每行
            lines = []
            for line in cleaned.split('\n'):
                line = line.strip()
                if line.startswith('*'):
                    line = line[1:].strip()
                if line and not self._is_meaningless(line):
                    lines.append(line)
            
            return '\n'.join(lines) if lines else ""
        
        return comment.strip()
    
    def _clean_docstring(self, docstring: str) -> str:
        """清理 docstring"""
        # 提取第一段（通常是最重要的描述）
        lines = []
        for line in docstring.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
            elif lines:  # 遇到空行，停止
                break
        
        result = ' '.join(lines)
        if len(result) > 500:  # 限制长度
            result = result[:500] + '...'
        
        return result
    
    def _is_meaningless(self, comment: str) -> bool:
        """判断注释是否无意义"""
        if not comment.strip():
            return True
        
        # 纯分隔符
        if re.match(r'^[-=_\s*]+$', comment):
            return True
        
        # 简单的 TODO/FIXME
        if re.match(r'^TODO[:\s]*$', comment, re.IGNORECASE):
            return True
        if re.match(r'^FIXME[:\s]*$', comment, re.IGNORECASE):
            return True
        
        # 短注释
        if len(comment.strip()) < 3:
            return True
        
        return False
    
    def _format_comments(self, file_path: Path, comments: List[str]) -> str:
        """格式化注释输出"""
        parts = [
            f"# File: {file_path.name}",
            "",
            "## Comments",
            "",
        ]
        
        for i, comment in enumerate(comments, 1):
            if '\n' in comment:
                parts.append(f"### Comment {i}")
                parts.append("")
                parts.append(comment)
                parts.append("")
            else:
                parts.append(f"{i}. {comment}")
        
        return '\n'.join(parts)


def extract_code_comments(file_path: Union[str, Path]) -> str:
    """便捷函数：提取代码注释"""
    extractor = CommentExtractor()
    return extractor.extract(file_path)