"""Command-line interface for arxiv_retriever."""

import os
import sys
from importlib.metadata import version as vsn
from typing import List, Optional

import httpx
import trio
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from typing_extensions import Annotated

from arxiv_retriever.fetcher import (
    download_from_links,
    fetch_papers,
    search_paper_by_title,
)
from arxiv_retriever.summary_util.exceptions import LLMProviderError
from arxiv_retriever.summary_util.pdf_extractor import (
    PDFExtractionError,
    build_pdf_prompt,
    collect_pdf_files,
    extract_text_from_pdf,
)
from arxiv_retriever.utils import process_papers

app = typer.Typer(no_args_is_help=True)
console = Console()



@app.command()
def fetch(
    categories: Annotated[
        List[str], typer.Argument(help="ArXiv categories to fetch papers from")
    ],
    limit: int = typer.Option(10, help="Maximum number of papers to fetch"),
    authors: Annotated[
        List[str],
        typer.Option(
            "--author",
            "-a",
            help="Author(s) to refine paper fetching by. Can be used multiple times.",
        ),
    ] = None,
    author_logic: str = typer.Option(
        "OR", "--author-logic", "-l", help="Logic to use for multiple authors: 'AND' or 'OR'"
    ),
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="LLM model for summarization. Format: provider:model_name "
            "(e.g., 'ollama:llama3', 'claude:claude-sonnet-4-6', 'gemini:gemini-3-flash-preview'). "
            "Or just the provider name: 'claude', 'gemini', 'ollama'.",
        ),
    ] = None,
):
    """
    Fetch papers from ArXiv based on categories, refined by options.

    :param categories: List of ArXiv categories to search
    :param limit: Total number of results to fetch
    :param authors: Optional list of author names to filter results by
    :param author_logic: Logic to use for multiple authors ('AND' or 'OR', default is 'OR')
    :param model: LLM model to use for summarization
    :return: None
    """
    author_logic = author_logic.upper()
    if author_logic not in ["AND", "OR"]:
        console.print(f"[yellow]Invalid author_logic: {author_logic}. Using default 'OR' logic.[/yellow]")
        author_logic = "OR"

    console.print(f"Fetching up to [bold]{limit}[/bold] papers from categories: [cyan]{', '.join(categories)}[/cyan]")
    if authors:
        console.print(f"Filtered by authors: [bold]{', '.join(authors)}[/bold] (using '{author_logic}' logic)...")

    try:
        papers = trio.run(fetch_papers, categories, limit, authors, author_logic)
        trio.run(process_papers, papers, model)
    except LLMProviderError as e:
        console.print(f"[bold red]LLM error:[/bold red] {e}")
    except httpx.HTTPError as e:
        console.print(f"[bold red]HTTP error occurred:[/bold red] {str(e)}")
    except trio.TooSlowError:
        console.print("[bold red]Operation timed out.[/bold red] Please try again later.")
    except KeyboardInterrupt:
        console.print("[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
        raise


@app.command()
def search(
    title: Annotated[str, typer.Argument(help="ArXiv title to search for")],
    limit: int = typer.Option(10, help="Maximum number of papers to search"),
    authors: Annotated[
        List[str],
        typer.Option(
            "--author",
            "-a",
            help="Author(s) to refine paper title search by. Can be used multiple times.",
        ),
    ] = None,
    author_logic: str = typer.Option(
        "OR", "--author-logic", "-l", help="Logic to use for multiple authors: 'AND' or 'OR'"
    ),
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="LLM model for summarization. Format: provider:model_name "
            "(e.g., 'ollama:llama3', 'claude:claude-sonnet-4-6', 'gemini:gemini-3-flash-preview'). "
            "Or just the provider name: 'claude', 'gemini', 'ollama'.",
        ),
    ] = None,
):
    """
    Search for papers on ArXiv using title, refined by options.

    :param title: Title of paper to search for
    :param limit: Total number of results to fetch
    :param authors: Optional list of author names to filter results by
    :param author_logic: Logic to use for multiple authors ('AND' or 'OR', default is 'OR')
    :param model: LLM model to use for summarization
    :return: None
    """
    author_logic = author_logic.upper()
    if author_logic not in ["AND", "OR"]:
        console.print(f"[yellow]Invalid author_logic: {author_logic}. Using default 'OR' logic.[/yellow]")
        author_logic = "OR"

    console.print(f"Searching for papers matching [bold]{title}[/bold]")
    if authors:
        console.print(f"Filtered by authors: [bold]{', '.join(authors)}[/bold] (using '{author_logic}' logic)...")

    try:
        papers = trio.run(search_paper_by_title, title, limit, authors, author_logic)
        trio.run(process_papers, papers, model)
    except LLMProviderError as e:
        console.print(f"[bold red]LLM error:[/bold red] {e}")
    except httpx.HTTPError as e:
        console.print(f"[bold red]HTTP error occurred:[/bold red] {str(e)}")
    except trio.TooSlowError:
        console.print("[bold red]Operation timed out.[/bold red] Please try again later.")
    except KeyboardInterrupt:
        console.print("[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
        raise


@app.command()
def download(
    links: Annotated[List[str], typer.Argument(help="ArXiv links to download")],
    download_dir: str = typer.Option(
        "./arxiv_downloads", "--download-dir", "-d", help="Directory to download papers"
    ),
):
    """
    Download papers from ArXiv using their links (PDF or abstract links).

    :param links: ArXiv links to download from
    :param download_dir: Directory to download papers
    :return: None
    """
    download_dir = typer.prompt("Enter download directory: ", default=download_dir)
    download_dir = os.path.expanduser(download_dir)

    console.print("[bold]Downloading papers from provided links...[/bold]")
    try:
        trio.run(download_from_links, links, download_dir)
        console.print(f"[green]Download complete. Papers saved to [bold]{download_dir}[/bold][/green]")
    except httpx.HTTPError as e:
        console.print(f"[bold red]HTTP error occurred:[/bold red] {str(e)}")
    except trio.TooSlowError:
        console.print("[bold red]Operation timed out.[/bold red] Please try again later.")
    except KeyboardInterrupt:
        console.print("[yellow]Operation cancelled by user.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {str(e)}")
        raise


@app.command()
def summarize(
    files: Annotated[
        List[str],
        typer.Argument(help="PDF file(s) or directory containing PDFs to summarize."),
    ],
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="LLM model for summarization. Format: provider:model_name "
            "(e.g., 'ollama:llama3', 'claude:claude-sonnet-4-6', 'gemini:gemini-3-flash-preview'). "
            "Or just the provider name: 'claude', 'gemini', 'ollama'. "
            "Default: ollama:llama3 (local, no rate limits).",
        ),
    ] = None,
    save: Annotated[
        bool,
        typer.Option("--save", "-s", help="Save summaries to a JSON file."),
    ] = False,
    delay: Annotated[
        int,
        typer.Option(
            "--delay",
            "-d",
            help="Delay in seconds between API requests (for rate limit safety).",
        ),
    ] = 3,
):
    """
    Summarize one or more PDF papers using an LLM provider.

    By default, Ollama (local) is used for summarization — no API rate limits
    or costs. You can also use cloud providers like Claude or Gemini, but ensure
    to set the appropriate environment variables.

    \b
    Examples:
        axiv summarize paper.pdf
        axiv summarize paper.pdf --model claude
        axiv summarize paper1.pdf paper2.pdf --model gemini
        axiv summarize ./arxiv_downloads/
        axiv summarize paper.pdf --model claude:claude-sonnet-4-6
        axiv summarize ./arxiv_downloads/ --model gemini:gemini-3-flash-preview --save

    \b
    Providers (use shorthand or provider:model_name):
        ollama  - Local models via Ollama (default: ollama:llama3, no rate limits)
        claude  - Anthropic Claude API (default: claude:claude-sonnet-4-6, requires ANTHROPIC_API_KEY)
        gemini  - Google Gemini API (default: gemini:gemini-3-flash-preview, requires GEMINI_API_KEY)
    """
    from arxiv_retriever.summary_util.config import (
        PROVIDER_DEFAULT_MODELS,
        get_default_model,
        parse_model_string,
    )
    from arxiv_retriever.summary_util.extractor import (
        get_default_summary_file,
        save_summaries_to_file,
    )
    from arxiv_retriever.summary_util.llm_interface import get_llm_response

    # Resolve which model is being used (parse to get canonical form)
    active_model = model or get_default_model()
    config = parse_model_string(active_model)
    provider_name = config.provider
    resolved_model = f"{config.provider}:{config.model_name}"

    # Inform user about the provider
    if provider_name == "ollama":
        console.print(
            Panel(
                f"[bold green]🏠 Ollama (local)[/bold green]: {resolved_model}\n"
                "No API rate limits — safe for batch processing.",
                title="Provider",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold cyan]☁️  Cloud provider[/bold cyan]: {resolved_model}\n"
                f"A {delay}-second delay will be applied between requests to avoid rate limits.",
                title="Provider",
                border_style="cyan",
            )
        )

    # Collect PDF files
    pdf_files = collect_pdf_files(files)

    if not pdf_files:
        console.print("[bold red]No PDF files found in the specified path(s).[/bold red]", style="red")
        raise typer.Exit(code=1)

    console.print(f"\n[bold]Found {len(pdf_files)} PDF file(s) to summarize:[/bold]\n")
    for i, f in enumerate(pdf_files, 1):
        console.print(f"  {i}. {os.path.basename(f)}")

    console.print(f"\n[bold]Summarizing {len(pdf_files)} paper(s)...[/bold]\n")

    results = []
    for i, pdf_path in enumerate(pdf_files):
        filename = os.path.basename(pdf_path)
        console.print(f"  [{i + 1}/{len(pdf_files)}] Processing: [bold]{filename}[/bold]")

        # Apply delay between requests (skip first)
        if i > 0 and provider_name != "ollama":
            console.print(f"    ⏳ Waiting {delay}s before next request...", style="dim")
            trio.run(_sleep, delay)

        try:
            text = extract_text_from_pdf(pdf_path)
            prompt = build_pdf_prompt(pdf_path, text)
            response = get_llm_response(prompt, model=resolved_model)

            results.append({
                "file": filename,
                "path": pdf_path,
                "extracted_info": response,
            })
            console.print("    ✅ Done", style="green")
        except PDFExtractionError as e:
            console.print(f"    ⚠️  Skipped: {e}", style="yellow")
        except LLMProviderError as e:
            console.print(f"    ❌ LLM error: {e}", style="red")

    # Display results using Rich panels and Markdown rendering
    if results:
        console.rule("[bold green]SUMMARIZATION COMPLETE[/bold green]")

        for info in results:
            # Render the LLM response as Markdown for better formatting
            summary_content = Markdown(info["extracted_info"])
            console.print(
                Panel(
                    summary_content,
                    title=f"📄 {info['file']}",
                    border_style="blue",
                    padding=(1, 2),
                )
            )

        # Save to file if requested
        if save:
            output_file = get_default_summary_file()
            save_summaries_to_file(results, output_file)
            console.print(f"\n✅ Summaries saved to: [bold]{output_file}[/bold]", style="green")

        console.print(
            f"\n📊 Summarized [bold]{len(results)}/{len(pdf_files)}[/bold] paper(s)."
        )
    else:
        console.print("\n⚠️  No papers were successfully summarized.", style="yellow")


async def _sleep(seconds: int):
    """Async sleep helper for trio.run()."""
    await trio.sleep(seconds)


@app.command()
def version():
    """
    Display version information for arxiv_retriever and core dependencies.

    :return: None
    """
    from rich.table import Table

    arxiv_retriever_version = vsn("arxiv_retriever")

    table = Table(title=f"arxiv_retriever v{arxiv_retriever_version}", border_style="blue")
    table.add_column("Component", style="bold")
    table.add_column("Version", style="cyan")

    table.add_row("arxiv_retriever", arxiv_retriever_version)
    table.add_row("Python", f"{sys.version_info.major}.{sys.version_info.minor}")
    table.add_row("Typer", vsn("typer"))
    table.add_row("Httpx", vsn("httpx"))
    table.add_row("Trio", vsn("trio"))

    console.print(table)


def main():
    """Entry point for arxiv_retriever"""
    app()


if __name__ == "__main__":
    main()
