import os
from typing import List, Dict
import typer
from openai import OpenAIError

from arxiv_retriever.summary_util.extractor import extract_essential_info
from arxiv_retriever.fetcher import download_papers


def extract_paper_metadata(papers: List[Dict]):
    """Extract metadata from papers in paper list."""
    for i, paper in enumerate(papers, 1):
        typer.echo(f"\n{i}. {paper['title']}")
        typer.echo(f"    Authors: {', '.join(paper['authors'])}")
        typer.echo(f"    Published: {paper['published']}")
        typer.echo(f"    Link to Abstract: {paper['abstract_link']}")
        typer.echo(f"    Link to PDF: {paper['pdf_link']}")
        typer.echo(f"    Summary: {paper['summary'][:100]}...") # TODO: possibly update to find index of first period character in summary then use for truncation. makes summary more complete.


def summarize_papers(papers: List[Dict]):
    """Summarize papers in paper list from their abstracts"""
    extracted_info = extract_essential_info(papers)
    for info in extracted_info:
        typer.echo(f"\n{info['title']}")
        typer.echo(f"Essential Information:\n{info['extracted_info']}")


async def process_papers(papers: List[Dict]):
    """
    Helper function to process retrieved papers

    :param papers: Papers to process
    :return: None
    """

    extract_paper_metadata(papers)

    if typer.confirm("\nWould you like to summarize these papers?"):
        try:
            summarize_papers(papers)
        except OpenAIError as e:
            typer.echo(e)

    if typer.confirm("\nWould you like to download these papers?"):
        default_dir = './arxiv_downloads'
        download_dir = typer.prompt(f"Enter download directory: ", default=default_dir)
        download_dir = os.path.expanduser(download_dir)  # expand user directory if used

        if not os.path.exists(download_dir):
            if typer.confirm(f"Directory {download_dir} does not exist. Create it?"):
                os.makedirs(download_dir)
            else:
                typer.echo("Download cancelled.")
                return

        await download_papers(papers, download_dir)
        typer.echo(f"Papers downloaded to {download_dir}")