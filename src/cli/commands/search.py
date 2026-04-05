"""搜索命令 - 支持健康检查和超时控制"""

from typing import Optional

import typer
from rich.console import Console

from src.cli.api_client import api_client, check_api_health, API_SEARCH_TIMEOUT
from src.cli.utils import truncate_text

app = typer.Typer(name="search", help="搜索")
console = Console()

# 全局变量控制是否跳过健康检查
_skip_health_check = False


def _perform_health_check() -> bool:
    """执行搜索前的健康检查
    
    Returns:
        True: 服务健康，可继续执行搜索
        False: 服务不健康，应终止搜索
    """
    health_result = check_api_health()
    
    if health_result["healthy"]:
        console.print(f"[dim]✓ RAG API 健康 ({health_result.get('response_time_ms', 0)}ms)[/dim]")
        return True
    else:
        console.print(f"[red]搜索前检查失败: {health_result['message']}[/red]")
        console.print(f"[dim]提示：使用 'ragctl service status' 查看服务状态[/dim]")
        console.print(f"[dim]提示：使用 --skip-health-check 跳过此检查（仅用于调试）[/dim]")
        return False


def _resolve_project_id(project: str) -> Optional[str]:
    """解析项目 ID（支持项目名称或 ID）"""
    # 如果是有效的 UUID 格式，直接返回
    if len(project) == 36 and project.count('-') == 4:
        return project
    
    # 否则按名称查找
    result = api_client.get("/api/v1/projects")
    if result and result.get("success"):
        data = result.get("data", {})
        projects = data if isinstance(data, list) else data.get("items", [])
        
        for p in projects:
            if isinstance(p, dict):
                # 精确匹配名称
                if p.get("name") == project:
                    return p.get("id")
                # 模糊匹配名称
                if project.lower() in (p.get("name", "")).lower():
                    return p.get("id")
    
    return project  # 找不到就原样返回，让 API 报错


def _search(
    project_id: str,
    query: str,
    search_mode: str,
    top_k: int,
    score_threshold: float,
    full_content: bool = False,
    skip_health_check: bool = False,
):
    """执行搜索
    
    Args:
        project_id: 项目 ID
        query: 查询内容
        search_mode: 搜索模式 (semantic/keyword/hybrid/hierarchical)
        top_k: 返回数量
        score_threshold: 分数阈值
        full_content: 是否显示完整内容
        skip_health_check: 是否跳过健康检查
    """
    # 健康检查（除非明确跳过）
    if not skip_health_check:
        if not _perform_health_check():
            console.print("[yellow]已终止搜索[/yellow]")
            raise typer.Exit(1)
    
    payload = {
        "project_id": project_id,
        "query": query,
        "top_k": top_k,
        "search_mode": search_mode,
        "score_threshold": score_threshold if score_threshold > 0 else None,
        "rerank": True,
    }
    
    # 使用搜索专用超时
    result = api_client.post("/api/v1/search", json_data=payload, timeout=API_SEARCH_TIMEOUT)
    
    if not result or not result.get("success"):
        console.print(f"[red]搜索失败: {result.get('message', '未知错误') if result else '无响应'}[/red]")
        return
    
    data = result.get("data", {})
    
    if isinstance(data, dict):
        results = data.get("results", [])
        query_time = data.get("query_time_ms", 0)
        total = data.get("total", len(results))
    else:
        results = data if isinstance(data, list) else []
        query_time = 0
        total = len(results)
    
    # 显示搜索信息
    console.print(f"\n[bold cyan]查询:[/bold cyan] {query}")
    console.print(f"[dim]模式: {search_mode} | 耗时: {query_time}ms | 结果: {total}[/dim]\n")
    
    if not results:
        console.print("[yellow]没有找到相关结果[/yellow]")
        return
    
    # 显示结果
    for i, r in enumerate(results, 1):
        if isinstance(r, dict):
            score = r.get("score", 0)
            content = r.get("content", "")
            doc_name = r.get("document_name", r.get("metadata", {}).get("filename", "-"))
            search_type = r.get("search_type", search_mode)
        else:
            score = 0
            content = str(r)
            doc_name = "-"
            search_type = search_mode
        
        # 分数颜色
        if score >= 0.8:
            score_color = "green"
        elif score >= 0.6:
            score_color = "yellow"
        else:
            score_color = "red"
        
        console.print(f"[bold cyan]{i}.[/bold cyan] [dim]({search_type}, 分数: [{score_color}]{score:.3f}[/{score_color}])[/dim]")
        console.print(f"   [magenta]{doc_name}[/magenta]")
        
        # 格式化内容（根据 full_content 决定是否截断）
        if full_content:
            display_content = content
        else:
            display_content = truncate_text(content, 300)
        console.print(f"   {display_content}\n")
        
        # 添加分隔线
        console.print("[dim]" + "─" * 60 + "[/dim]\n")


@app.command()
def semantic(
    project: str = typer.Argument(..., help="项目名称或ID"),
    query: str = typer.Argument(..., help="查询内容"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回数量"),
    score_threshold: float = typer.Option(0.0, "--threshold", "-t", help="分数阈值 (0-1)"),
    full: bool = typer.Option(False, "--full", "-f", help="显示完整内容"),
    skip_health_check: bool = typer.Option(False, "--skip-health-check", help="跳过健康检查（调试用）"),
):
    """语义搜索 - 基于向量相似度
    
    用法: ragctl search semantic <项目> <查询>
    示例: ragctl search semantic yunxi "采矿计划编制"
    """
    project_id = _resolve_project_id(project)
    if project_id != project:
        console.print(f"[dim]项目: {project} → {project_id[:8]}...[/dim]")
    _search(project_id, query, "semantic", top_k, score_threshold, full, skip_health_check)


@app.command()
def keyword(
    project: str = typer.Argument(..., help="项目名称或ID"),
    query: str = typer.Argument(..., help="查询内容"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回数量"),
    score_threshold: float = typer.Option(0.0, "--threshold", "-t", help="分数阈值 (0-1)"),
    full: bool = typer.Option(False, "--full", "-f", help="显示完整内容"),
    skip_health_check: bool = typer.Option(False, "--skip-health-check", help="跳过健康检查（调试用）"),
):
    """关键词搜索 - 基于 BM25
    
    用法: ragctl search keyword <项目> <查询>
    示例: ragctl search keyword yunxi "业务场景"
    """
    project_id = _resolve_project_id(project)
    if project_id != project:
        console.print(f"[dim]项目: {project} → {project_id[:8]}...[/dim]")
    _search(project_id, query, "keyword", top_k, score_threshold, full, skip_health_check)


@app.command()
def hybrid(
    project: str = typer.Argument(..., help="项目名称或ID"),
    query: str = typer.Argument(..., help="查询内容"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回数量"),
    score_threshold: float = typer.Option(0.0, "--threshold", "-t", help="分数阈值 (0-1)"),
    full: bool = typer.Option(False, "--full", "-f", help="显示完整内容"),
    skip_health_check: bool = typer.Option(False, "--skip-health-check", help="跳过健康检查（调试用）"),
):
    """混合搜索 - 向量 + BM25 + RRF 融合（推荐）
    
    用法: ragctl search hybrid <项目> <查询>
    示例: ragctl search hybrid yunxi "数据中台架构" -k 5
          ragctl search hybrid yunxi "数据中台架构" -k 1 --full
          ragctl search hybrid yunxi "数据中台架构" --skip-health-check（调试模式）
    """
    project_id = _resolve_project_id(project)
    if project_id != project:
        console.print(f"[dim]项目: {project} → {project_id[:8]}...[/dim]")
    _search(project_id, query, "hybrid", top_k, score_threshold, full, skip_health_check)


@app.command()
def hierarchical(
    project: str = typer.Argument(..., help="项目名称或ID"),
    query: str = typer.Argument(..., help="查询内容"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="返回数量"),
    score_threshold: float = typer.Option(0.0, "--threshold", "-t", help="分数阈值 (0-1)"),
    full: bool = typer.Option(False, "--full", "-f", help="显示完整内容"),
    skip_health_check: bool = typer.Option(False, "--skip-health-check", help="跳过健康检查（调试用）"),
):
    """层次化搜索 - RAPTOR（文档摘要 + chunks）
    
    用法: ragctl search hierarchical <项目> <查询>
    示例: ragctl search hierarchical yunxi "项目总体设计"
    
    特点: 先搜索文档摘要，再搜索相关 chunks，适合长文档
    """
    project_id = _resolve_project_id(project)
    if project_id != project:
        console.print(f"[dim]项目: {project} → {project_id[:8]}...[/dim]")
    _search(project_id, query, "hierarchical", top_k, score_threshold, full, skip_health_check)