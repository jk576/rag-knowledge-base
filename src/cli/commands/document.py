"""文档管理命令"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.cli.api_client import api_client
from src.cli.utils import format_size, confirm_action

app = typer.Typer(name="doc", help="文档管理")
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
        
        # 先尝试精确匹配
        for p in projects:
            if isinstance(p, dict) and p.get("name") == project:
                return p.get("id")
        
        # 再尝试模糊匹配（但优先返回最短的匹配，避免误匹配）
        matches = []
        for p in projects:
            if isinstance(p, dict):
                name = p.get("name", "")
                if project.lower() in name.lower():
                    matches.append((len(name), p.get("id")))
        
        # 返回名称最短的匹配（最精确）
        if matches:
            matches.sort()
            return matches[0][1]
    
    return project  # 找不到就原样返回，让 API 报错


@app.command("list")
def list_documents(
    project: str = typer.Argument(..., help="项目名称或ID"),
    limit: int = typer.Option(100, "--limit", "-l", help="每页返回数量（最大 500）"),
    page: int = typer.Option(1, "--page", "-p", help="页码（从 1 开始）"),
    search: str = typer.Option(None, "--search", "-s", help="文件名搜索"),
    full_id: bool = typer.Option(False, "--full", "-f", help="显示完整 ID"),
):
    """列出项目文档
    
    支持分页和文件名搜索：
    
    Examples:
        ragctl doc list yunxi                    # 显示前 100 个文档
        ragctl doc list yunxi -l 20             # 每页 20 个
        ragctl doc list yunxi -p 2              # 第 2 页
        ragctl doc list yunxi -s "调研"         # 搜索文件名包含"调研"
        ragctl doc list yunxi -s ".xlsx"        # 搜索所有 Excel 文件
    """
    # 参数验证
    if page < 1:
        console.print("[red]错误: 页码必须大于等于 1[/red]")
        raise typer.Exit(1)
    
    if limit < 1:
        console.print("[red]错误: 每页数量必须大于等于 1[/red]")
        raise typer.Exit(1)
    
    # 限制最大值，防止内存问题
    if limit > 500:
        console.print("[yellow]警告: 每页数量超过 500，已自动限制为 500[/yellow]")
        limit = 500
    
    # 解析项目名称为 ID
    project_id = _resolve_project_id(project)
    
    # 计算偏移量
    skip = (page - 1) * limit
    
    # 构建请求参数
    params = {"skip": skip, "limit": limit}
    if search:
        params["filename"] = search
    
    result = api_client.get(
        f"/api/v1/projects/{project_id}/documents",
        params=params,
    )
    
    if not result or not result.get("success"):
        console.print("[red]获取文档列表失败[/red]")
        return
    
    data = result.get("data", {})
    
    # 支持新旧两种格式
    if isinstance(data, dict):
        items = data.get("items", data.get("documents", []))
        total = data.get("total", len(items))
    else:
        items = data
        total = len(items)
    
    if not items:
        if search:
            console.print(f"[yellow]没有找到匹配 '{search}' 的文档[/yellow]")
        else:
            console.print("[yellow]项目中没有文档[/yellow]")
        return
    
    # 构建标题
    title = f"📄 文档列表"
    if search:
        title += f" (搜索: '{search}')"
    title += f" — 共 {total} 个"
    if total > len(items):
        title += f"，显示第 {page} 页 ({len(items)} 个)"
    
    table = Table(title=title)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("文件名", style="magenta")
    table.add_column("大小", justify="right")
    table.add_column("类型", style="green")
    table.add_column("分块", justify="right")
    table.add_column("状态", justify="center")
    
    for doc in items:
        doc_id = doc.get("id", "")
        filename = doc.get("filename", "-")
        file_size = doc.get("file_size", 0)
        doc_type = doc.get("doc_type", "unknown")
        chunk_count = doc.get("chunk_count", 0)
        status = doc.get("status", "unknown")
        
        # 状态显示
        if status == "completed":
            status_display = "✅ 完成"
        elif status == "failed":
            status_display = "❌ 失败"
        elif status == "processing":
            status_display = "⏳ 处理中"
        else:
            status_display = status
        
        # ID 显示格式
        if full_id:
            id_display = doc_id
        else:
            id_display = doc_id[:8] + "..." if len(doc_id) > 8 else doc_id
        
        table.add_row(
            id_display,
            filename[:40] + "..." if len(filename) > 40 else filename,
            format_size(file_size) if file_size else "-",
            doc_type,
            str(chunk_count),
            status_display,
        )
    
    console.print(table)
    
    # 显示分页提示
    if total > len(items):
        total_pages = (total + limit - 1) // limit
        console.print(f"\n[dim]提示: 共 {total_pages} 页，使用 -p {page+1} 查看下一页[/dim]")


@app.command()
def upload(
    project_id: str = typer.Argument(..., help="项目ID"),
    file_path: Path = typer.Argument(..., help="文件路径"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="递归处理目录"),
):
    """上传文档"""
    if not file_path.exists():
        console.print(f"[red]文件不存在: {file_path}[/red]")
        raise typer.Exit(1)
    
    files_to_upload = []
    
    if file_path.is_file():
        files_to_upload = [file_path]
    elif file_path.is_dir():
        pattern = "**/*" if recursive else "*"
        files_to_upload = [f for f in file_path.glob(pattern) if f.is_file()]
    
    if not files_to_upload:
        console.print("[yellow]没有找到要上传的文件[/yellow]")
        return
    
    console.print(f"[bold]准备上传 {len(files_to_upload)} 个文件...[/bold]\n")
    
    success_count = 0
    fail_count = 0
    
    for f in files_to_upload:
        console.print(f"  上传: {f.name}...", end=" ", style="cyan")
        
        result = api_client.upload_file(
            f"/api/v1/projects/{project_id}/documents",
            f,
        )
        
        if result and result.get("success"):
            console.print("[green]✓[/green]")
            success_count += 1
        else:
            console.print("[red]✗[/red]")
            fail_count += 1
    
    console.print(f"\n[bold]完成: {success_count} 成功, {fail_count} 失败[/bold]")


@app.command()
def delete(
    project_id: str = typer.Argument(..., help="项目ID"),
    doc_id: str = typer.Argument(..., help="文档ID"),
    force: bool = typer.Option(False, "--force", "-f", help="强制删除"),
):
    """删除文档"""
    if not force:
        if not confirm_action(f"确定要删除文档 {doc_id} 吗？"):
            console.print("[yellow]已取消[/yellow]")
            return
    
    result = api_client.delete(f"/api/v1/projects/{project_id}/documents/{doc_id}")
    
    if result and result.get("success"):
        console.print(f"[green]✓ 文档已删除[/green]")
    else:
        console.print(f"[red]✗ 删除失败: {result.get('message', '未知错误')}[/red]")


@app.command()
def export(
    project_id: str = typer.Argument(..., help="项目ID"),
    doc_id: str = typer.Argument(..., help="文档ID"),
    output: Path = typer.Option(None, "--output", "-o", help="输出文件路径"),
    format: str = typer.Option("txt", "--format", "-f", help="输出格式 (txt/markdown/json)"),
):
    """导出文档内容"""
    result = api_client.get(
        f"/api/v1/projects/{project_id}/documents/{doc_id}/export",
        params={"format": format},
    )
    
    if not result or not result.get("success"):
        console.print(f"[red]导出失败[/red]")
        return
    
    data = result.get("data", {})
    content = data.get("content", "") if isinstance(data, dict) else str(data)
    
    if output:
        output.write_text(content)
        console.print(f"[green]✓ 已保存到: {output}[/green]")
    else:
        console.print(content)
