"""Paper processing utilities."""

import os
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from arxiv_retriever.fetcher import download_papers
from arxiv_retriever.summary_util.exceptions import LLMProviderError
from arxiv_retriever.summary_util.extractor import (
    extract_essential_info_async,
    get_default_summary_file,
)

console = Console()


def extract_paper_metadata(papers: List[Dict]):
    """Extract metadata from papers in paper list."""
    for i, paper in enumerate(papers, 1):
        content = (
            f"[bold]Authors:[/bold] {', '.join(paper['authors'])}\n"
            f"[bold]Published:[/bold] {paper['published']}\n"
            f"[bold]Link to Abstract:[/bold] {paper['abstract_link']}\n"
            f"[bold]Link to PDF:[/bold] {paper['pdf_link']}\n"
            f"[bold]Summary:[/bold] {paper['summary'][:100]}..."
        )
        console.print(
            Panel(
                content,
                title=f"[bold]{i}. {paper['title']}[/bold]",
                border_style="blue",
                padding=(0, 2),
            )
        )


async def summarize_papers_async(
    papers: List[Dict],
    model: Optional[str] = None,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """
    Summarize papers asynchronously with rate limiting.

    Args:
        papers: List of paper dictionaries
        model: Optional model string (provider:model_name)
        output_file: Optional file path to save results

    Returns:
        List of extracted info dictionaries
    """
    total = len(papers)
    console.print(
        f"\n[bold]Summarizing {total} paper(s)[/bold] with 3-second delays between requests..."
    )
    console.print("[dim]This may take a while. Progress will be shown below.[/dim]\n")

    def progress_callback(current: int, total: int):
        console.print(f"  [{current}/{total}] Processed paper {current}", style="cyan")

    extracted_info = await extract_essential_info_async(
        papers,
        model=model,
        output_file=output_file,
        progress_callback=progress_callback,
    )

    # Display results
    console.rule("[bold green]SUMMARIZATION COMPLETE[/bold green]")

    for info in extracted_info:
        summary_content = Markdown(info["extracted_info"])
        header = (
            f"[bold]Authors:[/bold] {', '.join(info['authors'])}"
        )
        console.print(
            Panel(
                summary_content,
                title=f"📄 {info['title']}",
                subtitle=header,
                border_style="blue",
                padding=(1, 2),
            )
        )

    if output_file:
        console.print(f"\n✅ Summaries saved to: [bold]{output_file}[/bold]", style="green")

    return extracted_info


async def process_papers(papers: List[Dict], model: Optional[str] = None):
    """
    Helper function to process retrieved papers.

    Args:
        papers: Papers to process
        model: Optional model string for summarization (provider:model_name)
    """
    extract_paper_metadata(papers)

    if typer.confirm("\nWould you like to summarize these papers?"):
        # Ask if user wants to save to file
        save_to_file = typer.confirm(
            "Would you like to save summaries to a file?", default=True
        )
        output_file = None
        if save_to_file:
            default_file = get_default_summary_file()
            output_file = typer.prompt(
                "Enter output file path",
                default=default_file,
            )
            output_file = os.path.expanduser(output_file)

        try:
            await summarize_papers_async(papers, model=model, output_file=output_file)
        except LLMProviderError as e:
            console.print(f"[bold red]Summarization error:[/bold red] {e}")

    if typer.confirm("\nWould you like to download these papers?"):
        default_dir = "./arxiv_downloads"
        download_dir = typer.prompt("Enter download directory: ", default=default_dir)
        download_dir = os.path.expanduser(download_dir)

        if not os.path.exists(download_dir):
            if typer.confirm(f"Directory {download_dir} does not exist. Create it?"):
                os.makedirs(download_dir)
            else:
                console.print("[yellow]Download cancelled.[/yellow]")
                return

        await download_papers(papers, download_dir)
        console.print(f"[green]Papers downloaded to [bold]{download_dir}[/bold][/green]")
