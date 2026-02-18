"""Extract essential information from papers using LLM."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import trio

from arxiv_retriever.summary_util.llm_interface import get_llm_response

# Delay between LLM requests to avoid rate limiting (in seconds)
LLM_REQUEST_DELAY = 3


def _build_prompt(paper: Dict) -> str:
    """Build the extraction prompt for a paper."""
    return f"""
Title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Summary: {paper['summary']}


Please extract and summarize the most essential information from this paper abstract.
Focus on the main contributions, key findings, and potential impact of the research.
Suggest future research directions that is grounded in factual and currently available research.
Limit your response to 3-5 concise bullet points.
"""


async def _extract_single_paper(
    paper: Dict,
    model: Optional[str],
    results: List[Dict],
    index: int,
    progress_callback=None,
) -> None:
    """
    Extract essential info from a single paper with rate limiting.

    Args:
        paper: Paper dictionary
        model: Model string
        results: Shared list to store results
        index: Index for ordering results
        progress_callback: Optional callback for progress updates
    """
    # Add delay before request to avoid overwhelming the API
    if index > 0:
        await trio.sleep(LLM_REQUEST_DELAY)

    prompt = _build_prompt(paper)
    response = get_llm_response(prompt, model=model)

    results.append({
        "index": index,
        "title": paper["title"],
        "authors": paper["authors"],
        "extracted_info": response,
    })

    if progress_callback:
        progress_callback(index + 1, len(paper))


async def extract_essential_info_async(
    papers: List[Dict],
    model: Optional[str] = None,
    output_file: Optional[str] = None,
    progress_callback=None,
) -> List[Dict]:
    """
    Extract essential information from papers asynchronously with rate limiting.

    Papers are processed sequentially with delays to avoid API rate limits.

    Args:
        papers: List of paper dictionaries with title, authors, summary
        model: Optional model string (provider:model_name)
        output_file: Optional file path to save results
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of dictionaries with title and extracted_info
    """
    results = []

    # Process papers sequentially with delays
    for i, paper in enumerate(papers):
        if i > 0:
            await trio.sleep(LLM_REQUEST_DELAY)

        prompt = _build_prompt(paper)
        response = get_llm_response(prompt, model=model)

        results.append({
            "title": paper["title"],
            "authors": paper["authors"],
            "extracted_info": response,
        })

        if progress_callback:
            progress_callback(i + 1, len(papers))

    # Save to file if requested
    if output_file:
        save_summaries_to_file(results, output_file)

    return results


def extract_essential_info(
    papers: List[Dict], model: Optional[str] = None
) -> List[Dict]:
    """
    Extract essential information from papers (sync wrapper).

    Args:
        papers: List of paper dictionaries with title, authors, summary
        model: Optional model string (provider:model_name)

    Returns:
        List of dictionaries with title and extracted_info
    """
    return trio.run(extract_essential_info_async, papers, model)


def save_summaries_to_file(summaries: List[Dict], output_file: str) -> None:
    """
    Save summaries to a JSON file.

    Args:
        summaries: List of summary dictionaries
        output_file: Path to output file
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_data = {
        "generated_at": datetime.now().isoformat(),
        "count": len(summaries),
        "summaries": summaries,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def get_default_summary_file() -> str:
    """Get the default summary output file path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"./arxiv_summaries/summaries_{timestamp}.json"
