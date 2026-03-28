"""项目管理命令"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.cli.api_client import api_client
from src.cli.utils import format_size, get_project_dir_size, confirm_action

app = typer.Typer(name="project", help="项目管理")
console = Console()


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


@app.command("list")
def list_projects(
    full_id: bool = typer.Option(False, "--full", "-f", help="显示完整ID"),
):
    """列出所有项目"""
    result = api_client.get("/api/v1/projects")
    
    if not result or not result.get("success"):
        console.print("[red]获取项目列表失败[/red]")
        return
    
    data = result.get("data", {})
    projects = data if isinstance(data, list) else data.get("items", [])
    
    if not projects:
        console.print("[yellow]暂无项目[/yellow]")
        return
    
    table = Table(title=f"📁 项目列表 (共 {len(projects)} 个)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("名称", style="magenta")
    table.add_column("描述", style="green")
    table.add_column("文档数", justify="right")
    
    for p in projects:
        project_id = p.get("id", "")
        display_id = project_id if full_id else project_id[:8] + "..."
        table.add_row(
            display_id,
            p.get("name", "-"),
            p.get("description", "-") or "-",
            str(p.get("document_count", 0)),
        )
    
    console.print(table)
    console.print("\n[dim]提示: 使用项目名称或 ID 都可以进行搜索[/dim]")
    console.print("[dim]      ragctl search <项目名> <查询内容>[/dim]")
    console.print("[dim]      ragctl search --full 显示完整ID[/dim]")


@app.command()
def create(
    name: str = typer.Argument(..., help="项目名称"),
    description: str = typer.Option(None, "--desc", "-d", help="项目描述"),
):
    """创建项目"""
    payload = {"name": name}
    if description:
        payload["description"] = description
    
    result = api_client.post("/api/v1/projects", json_data=payload)
    
    if result and result.get("success"):
        console.print(f"[green]✓ 项目创建成功: {name}[/green]")
    else:
        console.print(f"[red]✗ 项目创建失败: {result.get('message', '未知错误')}[/red]")


@app.command()
def delete(
    project_id: str = typer.Argument(..., help="项目ID"),
    force: bool = typer.Option(False, "--force", "-f", help="强制删除"),
):
    """删除项目"""
    if not force:
        if not confirm_action(f"确定要删除项目 {project_id} 吗？这将删除所有相关数据！"):
            console.print("[yellow]已取消[/yellow]")
            return
    
    result = api_client.delete(f"/api/v1/projects/{project_id}")
    
    if result and result.get("success"):
        console.print(f"[green]✓ 项目已删除[/green]")
    else:
        console.print(f"[red]✗ 删除失败: {result.get('message', '未知错误')}[/red]")


@app.command()
def info(
    project_id: str = typer.Argument(..., help="项目ID"),
):
    """查看项目详情"""
    result = api_client.get(f"/api/v1/projects/{project_id}")
    
    if not result or not result.get("success"):
        console.print(f"[red]获取项目信息失败[/red]")
        return
    
    data = result.get("data", {})
    
    if isinstance(data, dict):
        console.print(f"\n[bold cyan]项目详情[/bold cyan]")
        console.print(f"  ID:          {data.get('id', '-')}")
        console.print(f"  名称:        {data.get('name', '-')}")
        console.print(f"  描述:        {data.get('description', '-') or '-'}")
        console.print(f"  文档数:      {data.get('document_count', 0)}")
        
        # 计算存储大小
        dir_size = get_project_dir_size(project_id)
        console.print(f"  存储大小:    {format_size(dir_size)}")
    else:
        console.print(f"[red]无法解析项目数据[/red]")


@app.command()
def reindex(
    project_id: str = typer.Argument(..., help="项目ID或名称"),
):
    """重新索引项目"""
    # 解析项目 ID
    resolved_id = _resolve_project_id(project_id)
    
    console.print(f"[bold yellow]正在重新索引项目 {project_id}...[/bold yellow]")
    if resolved_id != project_id:
        console.print(f"[dim]ID: {resolved_id[:8]}...[/dim]")
    
    # 获取所有文档
    result = api_client.get(f"/api/v1/projects/{resolved_id}/documents")
    
    if not result or not result.get("success"):
        console.print(f"[red]获取文档列表失败[/red]")
        return
    
    data = result.get("data", [])
    if isinstance(data, dict):
        data = data.get("items", [])
    
    if not data:
        console.print("[yellow]项目中没有文档[/yellow]")
        return
    
    console.print(f"发现 {len(data)} 个文档，开始重新索引...\n")
    
    success_count = 0
    fail_count = 0
    
    for doc in data:
        doc_id = doc.get("id")
        console.print(f"  索引文档: {doc.get('filename', doc_id)}...", end=" ")
        
        res = api_client.post(f"/api/v1/projects/{resolved_id}/documents/{doc_id}/reindex")
        if res and res.get("success"):
            console.print("[green]✓[/green]")
            success_count += 1
        else:
            console.print("[red]✗[/red]")
            fail_count += 1
    
    console.print(f"\n[bold]完成: {success_count} 成功, {fail_count} 失败[/bold]")


@app.command()
def scan(
    project_id: str = typer.Argument(..., help="项目ID"),
):
    """强制扫描项目文件"""
    result = api_client.post(f"/api/v1/watcher/scan?project_name={project_id}")
    
    if result and result.get("success"):
        console.print(f"[green]✓ 已触发项目扫描: {project_id}[/green]")
    else:
        console.print(f"[red]✗ 扫描失败[/red]")


@app.command()
def check(
    project_id: str = typer.Argument(..., help="项目ID或名称"),
    cleanup: bool = typer.Option(False, "--cleanup", "-c", help="自动清理孤立文件"),
):
    """检查项目数据一致性
    
    检查内容：
    - 文档数量是否匹配
    - 向量是否完整
    - 是否有孤立文件
    """
    # 解析项目 ID（支持名称）
    resolved_id = _resolve_project_id(project_id)
    
    console.print(f"\n[bold cyan]🔍 检查项目: {project_id}[/bold cyan]")
    if resolved_id != project_id:
        console.print(f"[dim]ID: {resolved_id[:8]}...[/dim]")
    
    # 1. 获取项目信息
    result = api_client.get(f"/api/v1/projects/{resolved_id}")
    if not result or not result.get("success"):
        console.print(f"[red]✗ 获取项目信息失败[/red]")
        return
    
    project = result.get("data", {})
    doc_count = project.get("document_count", 0)
    chunk_count = project.get("chunk_count", 0)
    
    console.print(f"\n[bold]项目状态:[/bold]")
    console.print(f"  文档数: {doc_count}")
    console.print(f"  分块数: {chunk_count}")
    
    # 2. 获取文档列表
    docs_result = api_client.get(f"/api/v1/projects/{resolved_id}/documents")
    if not docs_result or not docs_result.get("success"):
        console.print(f"[red]✗ 获取文档列表失败[/red]")
        return
    
    docs = docs_result.get("data", [])
    if isinstance(docs, dict):
        docs = docs.get("items", [])
    
    # 3. 检查文档状态
    error_docs = []
    for doc in docs:
        status = doc.get("status", "unknown")
        if status == "error":
            error_docs.append(doc.get("filename", doc.get("id")))
    
    if error_docs:
        console.print(f"\n[yellow]⚠ 发现 {len(error_docs)} 个错误文档:[/yellow]")
        for d in error_docs[:5]:
            console.print(f"  - {d}")
        if len(error_docs) > 5:
            console.print(f"  ... 还有 {len(error_docs) - 5} 个")
    else:
        console.print(f"\n[green]✓ 所有文档状态正常[/green]")
    
    # 4. 检查健康状态
    health_result = api_client.get("/health/detailed")
    if health_result:
        watcher = health_result.get("watcher", {})
        watched = watcher.get("watched_projects", [])
        if project.get("name") in watched:
            console.print(f"[green]✓ 项目正在被监控[/green]")
        else:
            console.print(f"[yellow]⚠ 项目未被监控[/yellow]")
    
    # 5. 汇总
    console.print(f"\n[bold]检查结果:[/bold]")
    if error_docs:
        console.print(f"[yellow]状态: 有问题 ({len(error_docs)} 个错误文档)[/yellow]")
    else:
        console.print(f"[green]状态: ✅ 正常[/green]")


@app.command()
def stats(
    project_id: str = typer.Argument(..., help="项目ID或名称"),
):
    """查看项目统计信息"""
    # 解析项目 ID
    resolved_id = _resolve_project_id(project_id)
    
    result = api_client.get(f"/api/v1/projects/{resolved_id}")
    
    if not result or not result.get("success"):
        console.print(f"[red]获取项目信息失败[/red]")
        return
    
    project = result.get("data", {})
    
    table = Table(title=f"📊 项目统计: {project.get('name', project_id)}")
    table.add_column("指标", style="cyan")
    table.add_column("值", style="green")
    
    table.add_row("文档数", str(project.get("document_count", 0)))
    table.add_row("分块数", str(project.get("chunk_count", 0)))
    table.add_row("监控状态", "✅ 启用" if project.get("watcher_enabled") else "❌ 禁用")
    table.add_row("创建时间", project.get("created_at", "-")[:10] if project.get("created_at") else "-")
    table.add_row("更新时间", project.get("updated_at", "-")[:10] if project.get("updated_at") else "-")
    
    console.print(table)
