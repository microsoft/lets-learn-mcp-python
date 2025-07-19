"""
Console and Rich utility functions for MCP tutorial labs.
This module provides consistent console output formatting across all MCP lab exercises.
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
import time
import asyncio

# Initialize the global console
console = Console()

def display_header(title: str, subtitle: str = None):
    """Display a formatted header for lab exercises."""
    console.print()
    console.print(Panel(
        f"[bold blue]{title}[/]" + (f"\n[yellow]{subtitle}[/]" if subtitle else ""),
        title="🔧 MCP Tutorial",
        title_align="left",
        border_style="blue",
        padding=(1, 2)
    ))
    console.print()

def display_info_panel(content: str, title: str = "ℹ️ Information", style: str = "cyan"):
    """Display information in a styled panel."""
    console.print(Panel(
        Markdown(content),
        title=title,
        title_align="left",
        border_style=style,
        padding=(1, 2),
        expand=False
    ))
    console.print()

def display_success_panel(content: str, title: str = "✅ Success"):
    """Display success message in a green panel."""
    console.print(Panel(
        Markdown(content),
        title=title,
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False
    ))
    console.print()

def display_error_panel(content: str, title: str = "❌ Error"):
    """Display error message in a red panel."""
    console.print(Panel(
        Markdown(content),
        title=title,
        title_align="left",
        border_style="red",
        padding=(1, 2),
        expand=False
    ))
    console.print()

def display_code_panel(code: str, language: str = "python", title: str = "📝 Code Example"):
    """Display code in a syntax-highlighted panel."""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(Panel(
        syntax,
        title=title,
        title_align="left",
        border_style="yellow",
        padding=(1, 2),
        expand=False
    ))
    console.print()

def display_step(step_number: int, title: str, description: str = None):
    """Display a numbered step in the tutorial."""
    step_text = f"[bold blue]Step {step_number}:[/] [bold]{title}[/]"
    if description:
        step_text += f"\n{description}"
    
    console.print(Panel(
        step_text,
        title=f"📋 Step {step_number}",
        title_align="left",
        border_style="magenta",
        padding=(1, 2),
        expand=False
    ))
    console.print()

def prompt_user(message: str, default: str = None) -> str:
    """Get user input with rich formatting."""
    return Prompt.ask(f"[bold green]{message}[/]", default=default)

def prompt_continue(message: str = "Press Enter to continue..."):
    """Pause execution and wait for user to continue."""
    Prompt.ask(f"[dim]{message}[/]", default="")

def show_progress(description: str, duration: float = 2.0):
    """Show a progress spinner for the given duration."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=None)
        time.sleep(duration)

def create_table(title: str, headers: list, rows: list) -> Table:
    """Create a rich table with the given headers and rows."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    for header in headers:
        table.add_column(header, style="cyan")
    
    for row in rows:
        table.add_row(*row)
    
    return table

def display_table(title: str, headers: list, rows: list):
    """Display a table with rich formatting."""
    table = create_table(title, headers, rows)
    console.print(table)
    console.print()

def display_json_data(data: dict, title: str = "📊 JSON Data"):
    """Display JSON data in a formatted panel."""
    import json
    json_str = json.dumps(data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    console.print(Panel(
        syntax,
        title=title,
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False
    ))
    console.print()

def section_separator():
    """Print a visual separator between sections."""
    console.print("\n" + "─" * 80 + "\n")

def lab_complete():
    """Display lab completion message."""
    console.print(Panel(
        "[bold green]🎉 Lab Complete![/]\n\nYou have successfully completed this lab exercise.",
        title="✅ Congratulations",
        title_align="center",
        border_style="green",
        padding=(1, 2)
    ))

async def async_show_progress(description: str, duration: float = 2.0):
    """Async version of show_progress."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=None)
        await asyncio.sleep(duration)
