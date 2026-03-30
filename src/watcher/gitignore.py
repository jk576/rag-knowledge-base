""".gitignore 解析器

使用 pathspec 库解析 .gitignore 文件，支持文件变更检测时的忽略规则。
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

# 从统一配置导入文件类型定义
from src.core.comment_extractor import (
    DOC_EXTENSIONS,
    CODE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
)

try:
    import pathspec
    PATHSPEC_AVAILABLE = True
except ImportError:
    PATHSPEC_AVAILABLE = False
    logging.warning("pathspec not installed, .gitignore support will be limited")

logger = logging.getLogger(__name__)


class GitIgnoreParser:
    """.gitignore 解析器
    
    支持解析项目根目录的 .gitignore 文件，并提供文件匹配检查功能。
    同时包含全局忽略规则（如 .git/, node_modules/ 等）。
    """
    
    # 全局忽略规则（始终生效，优先级最高）
    GLOBAL_IGNORE_PATTERNS = [
        ".git/",
        ".git\\",
        "node_modules/",
        "node_modules\\",
        "**/node_modules/",
        "**/node_modules\\",
        "__pycache__/",
        "__pycache__\\",
        "*.pyc",
        ".DS_Store",
        "dist/",
        "dist\\",
        "build/",
        "build\\",
        ".venv/",
        ".venv\\",
        "venv/",
        "venv\\",
        ".env/",
        ".env\\",
        ".idea/",
        ".idea\\",
        ".vscode/",
        ".vscode\\",
        "*.log",
        "*.tmp",
        "*.temp",
        "*.swp",
        "*.swo",
        "*~",
        ".pytest_cache/",
        ".pytest_cache\\",
        ".mypy_cache/",
        ".mypy_cache\\",
        ".coverage",
        "htmlcov/",
        "htmlcov\\",
    ]
    
    def __init__(self, project_path: Union[str, Path]):
        """
        初始化 GitIgnore 解析器
        
        Args:
            project_path: 项目根目录路径
        """
        self.project_path = Path(project_path).resolve()
        self.gitignore_path = self.project_path / ".gitignore"
        self.spec: Optional[pathspec.PathSpec] = None
        self._last_modified: float = 0
        self._load_gitignore()
    
    def _load_gitignore(self) -> None:
        """加载 .gitignore 文件内容"""
        if not PATHSPEC_AVAILABLE:
            logger.debug("pathspec not available, skipping .gitignore loading")
            return
        
        if not self.gitignore_path.exists():
            logger.debug(f"No .gitignore found at {self.gitignore_path}")
            self.spec = pathspec.PathSpec.from_lines("gitwildmatch", [])
            return
        
        try:
            with open(self.gitignore_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            
            # 过滤空行和注释
            patterns = [line for line in lines if line.strip() and not line.strip().startswith("#")]
            
            self.spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
            self._last_modified = self.gitignore_path.stat().st_mtime
            
            logger.debug(f"Loaded {len(patterns)} patterns from {self.gitignore_path}")
        except Exception as e:
            logger.error(f"Error loading .gitignore: {e}")
            self.spec = pathspec.PathSpec.from_lines("gitwildmatch", [])
    
    def reload_if_changed(self) -> bool:
        """
        如果 .gitignore 文件有变更，重新加载
        
        Returns:
            是否进行了重新加载
        """
        if not self.gitignore_path.exists():
            return False
        
        try:
            current_mtime = self.gitignore_path.stat().st_mtime
            if current_mtime > self._last_modified:
                logger.info(f".gitignore changed, reloading...")
                self._load_gitignore()
                return True
        except Exception as e:
            logger.error(f"Error checking .gitignore modification: {e}")
        
        return False
    
    def is_ignored(self, file_path: Union[str, Path]) -> bool:
        """
        检查文件是否被忽略
        
        检查顺序：
        1. 全局忽略规则（.git/, node_modules/ 等）
        2. 项目 .gitignore 中的规则
        
        Args:
            file_path: 文件路径（可以是绝对路径或相对路径）
            
        Returns:
            是否被忽略
        """
        file_path = Path(file_path)
        
        # 转换为相对于项目根目录的路径（用于匹配）
        try:
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.project_path)
            else:
                rel_path = file_path
                file_path = self.project_path / file_path
        except ValueError:
            # 路径不在项目目录下
            rel_path = file_path
        
        rel_path_str = str(rel_path).replace("\\", "/")
        rel_path_str_dir = rel_path_str + "/" if not rel_path_str.endswith("/") else rel_path_str
        
        # 1. 检查全局忽略规则
        for pattern in self.GLOBAL_IGNORE_PATTERNS:
            pattern = pattern.replace("\\", "/")
            if self._match_pattern(rel_path_str, pattern) or self._match_pattern(rel_path_str_dir, pattern):
                logger.debug(f"File {rel_path} ignored by global pattern: {pattern}")
                return True
        
        # 2. 检查 .gitignore 规则
        if self.spec and PATHSPEC_AVAILABLE:
            if self.spec.match_file(rel_path_str):
                logger.debug(f"File {rel_path} ignored by .gitignore")
                return True
        
        return False
    
    def _match_pattern(self, path: str, pattern: str) -> bool:
        """
        简单的模式匹配
        
        支持以下模式：
        - 以 / 结尾的目录匹配（匹配路径中的任何位置）
        - **/ 前缀匹配任意层级目录
        - 通配符匹配 (*, ?)
        
        Args:
            path: 文件路径
            pattern: 匹配模式
            
        Returns:
            是否匹配
        """
        import fnmatch
        
        # 处理 **/ 前缀（匹配任意层级）
        if pattern.startswith("**/"):
            # 移除 **/ 前缀，然后检查路径的任何部分是否匹配
            sub_pattern = pattern[3:]  # 去掉 **/
            
            # 目录匹配（以 / 结尾）
            if sub_pattern.endswith("/"):
                dir_pattern = sub_pattern.rstrip("/")
                # 检查路径的任何部分是否以该目录开头或等于该目录
                path_parts = path.split("/")
                for i, part in enumerate(path_parts):
                    if part == dir_pattern:
                        return True
                    # 也检查路径片段是否以 dir_pattern/ 开头
                    path_prefix = "/".join(path_parts[:i+1])
                    if path_prefix.endswith(dir_pattern) or f"/{dir_pattern}" in "/" + path_prefix:
                        return True
                return False
            
            # 非目录模式：匹配路径的任何部分
            if fnmatch.fnmatch(path, sub_pattern):
                return True
            # 检查路径的任何部分
            for part in path.split("/"):
                if fnmatch.fnmatch(part, sub_pattern):
                    return True
            return False
        
        # 目录匹配（以 / 结尾）
        if pattern.endswith("/"):
            dir_pattern = pattern.rstrip("/")
            # 检查路径是否包含该目录
            if f"/{dir_pattern}/" in "/" + path + "/":
                return True
            if path.startswith(dir_pattern + "/") or path == dir_pattern:
                return True
            # 检查路径的任何部分是否匹配该目录名
            for part in path.split("/"):
                if part == dir_pattern:
                    # 确保这是一个目录边界匹配
                    idx = path.find(part)
                    if idx >= 0:
                        after = path[idx + len(part):]
                        if after.startswith("/") or after == "":
                            return True
            return False
        
        # 通配符匹配
        if fnmatch.fnmatch(path, pattern):
            return True
        
        # 匹配路径的任何部分
        if "/" not in pattern:  # 没有 / 的模式匹配任何层级
            for part in path.split("/"):
                if fnmatch.fnmatch(part, pattern):
                    return True
        
        return False
    
    def should_process(self, file_path: Union[str, Path]) -> bool:
        """
        检查文件是否应该被处理（不被忽略且是支持的格式）
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否应该处理
        """
        file_path = Path(file_path)
        
        # 检查是否被忽略
        if self.is_ignored(file_path):
            return False
        
        # 只处理文件，不处理目录
        if file_path.is_dir():
            return False
        
        # 检查是否是支持的格式（使用统一配置）
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(f"File {file_path} has unsupported extension: {ext}")
            return False
        
        return True


class GitIgnoreCache:
    """.gitignore 解析器缓存
    
    为每个项目缓存 GitIgnoreParser 实例，避免重复解析。
    """
    
    def __init__(self):
        self._cache: dict[str, GitIgnoreParser] = {}
    
    def get_parser(self, project_path: Union[str, Path]) -> GitIgnoreParser:
        """
        获取或创建 GitIgnoreParser
        
        Args:
            project_path: 项目路径
            
        Returns:
            GitIgnoreParser 实例
        """
        path_key = str(Path(project_path).resolve())
        
        if path_key not in self._cache:
            self._cache[path_key] = GitIgnoreParser(project_path)
        else:
            # 检查是否需要重新加载
            self._cache[path_key].reload_if_changed()
        
        return self._cache[path_key]
    
    def invalidate(self, project_path: Union[str, Path]) -> None:
        """
        使缓存失效
        
        Args:
            project_path: 项目路径
        """
        path_key = str(Path(project_path).resolve())
        if path_key in self._cache:
            del self._cache[path_key]
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()


# 全局缓存实例
gitignore_cache = GitIgnoreCache()
