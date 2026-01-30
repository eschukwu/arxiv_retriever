import os
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET  # TODO: explore way to parse XML data more securely
import urllib.parse

import typer
import trio
import httpx

WAIT_TIME = 3  # number of seconds to wait between calls
BASE_URL = "https://export.arxiv.org/api/query?"

async def rate_limited_get(client: httpx.AsyncClient, url: str) -> httpx.Response:
    """Make an asynchronous GET request with rate limiting."""
    await trio.sleep(WAIT_TIME)
    response = await client.get(url)
    return response


async def fetch_papers(categories: List[str], limit: int, authors: Optional[List[str]] = None, author_logic: str = 'OR') -> List[Dict]:
    """
    Fetch papers from ArXiv using given categories and limit, with optional author filter.

    :param categories: List of ArXiv categories to search
    :param limit: Total number of results to fetch
    :param authors: Optional list of author names to filter results by
    :param author_logic: Logic to use for multiple authors ('AND' or 'OR', default is 'AND')
    :return: List of dictionaries containing paper information
    """
    papers = []
    start = 0  # index of the first returned result
    max_results_per_query = 100

    category_query = '+OR+'.join(f'cat:{cat}' for cat in categories)

    if authors:
        author_join = '+AND+' if author_logic.upper() == 'AND' else '+OR+'
        author_query = '+AND+(' + author_join.join(f'au:"{urllib.parse.quote_plus(author)}"' for author in authors) + ')'
    else:
        author_query = ''

    async with httpx.AsyncClient() as client:
        while start < limit:
            query = f"search_query={category_query}{author_query}&sortBy=submittedDate&sortOrder=descending&start={start}&max_results={max_results_per_query}"
            response = await rate_limited_get(client, BASE_URL + query)

            if response.status_code == 200:
                papers.extend(parse_arxiv_response(response.text))
                start += max_results_per_query
            else:
                raise Exception(f"Failed to fetch papers: HTTP {response.status_code}")

    return papers[:limit]  # Trim to the requested number of results


async def search_paper_by_title(title: str, limit: int, authors: Optional[List[str]] = None, author_logic: str = 'OR') -> List[Dict]:
    """
    Search for papers on ArXiv using title, optionally filtered by author and return `limit` papers.

    :param title: Title of paper to search for
    :param limit: Total number of results to fetch
    :param authors: Optional list of author names to filter results by
    :param author_logic: Logic to use for multiple authors ('AND' or 'OR', default is 'AND')
    :return: List of dictionaries containing paper information
    """
    encoded_title = urllib.parse.quote_plus(title)
    papers = []
    start = 0
    max_results_per_query = 100

    title_query = f'ti:"{encoded_title}"'

    if authors:
        author_join = '+AND+' if author_logic.upper() == 'AND' else '+OR+'
        author_query = '+AND+(' + author_join.join(f'au:"{urllib.parse.quote_plus(author)}"' for author in authors) + ')'
    else:
        author_query = ''

    async with httpx.AsyncClient() as client:
        while start < limit:
            query = f"search_query={title_query}{author_query}&sortBy=relevance&sortOrder=descending&start={start}&max_results={max_results_per_query}"
            response = await rate_limited_get(client, BASE_URL + query)

            if response.status_code == 200:
                papers.extend(parse_arxiv_response(response.text))
                start += max_results_per_query
            else:
                raise Exception(f"Failed to search papers: HTTP {response.status_code}")

    return papers[:limit]


def parse_arxiv_response(xml_data: str) -> List[Dict]:
    """
    Parse arXiv XML response and return paper information

    :param xml_data: XML response from arXiv API
    :return: List of dictionaries containing paper information
    """
    root = ET.fromstring(xml_data)
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall('atom:entry', namespace):
        # find pdf link
        pdf_link = next((link.get('href') for link in entry.findall('atom:link', namespace) if link.get('title') == 'pdf'), None)
        paper = {
            'title': entry.find('atom:title', namespace).text.strip(),
            'authors': [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)],
            'summary': entry.find('atom:summary', namespace).text.strip(),
            'published': entry.find('atom:published', namespace).text.strip(),
            'abstract_link': entry.find('atom:id', namespace).text.strip(),
            'pdf_link': pdf_link,
        }
        papers.append(paper)

    return papers


async def _download_paper(client: httpx.AsyncClient, paper: Dict, download_dir: str):
    """
    Download a paper and save it to the specified directory.

    :param client: httpx AsyncClient instance
    :param paper: Dictionary containing paper information
    :param download_dir: Directory to save downloaded papers
    :return: None
    """
    if not paper['pdf_link']:
        typer.echo(f"No PDF link for paper {paper['title']}")
        return

    filename = f"{paper['title'].replace(' ', '_')[:20]}.pdf"
    filepath = os.path.join(download_dir, filename)

    try:
        response = await client.get(paper['pdf_link'])
        response.raise_for_status()
        with open(filepath, 'wb') as file:
            file.write(response.content)
        typer.echo(f"Downloaded {filename} to {filepath}")
    except httpx.HTTPStatusError as err:
        typer.echo(f"Failed to download '{paper['title']}': HTTP {err.response.status_code}")
    except Exception as e:
        typer.echo(f"An error occurred while downloading '{paper['title']}': {str(e)}")


async def download_papers(papers: List[Dict], download_dir: str):
    """
    Download multiple papers asynchronously and save them to the specified directory.

    :param papers: List of dictionaries containing paper information
    :param download_dir: Directory to save downloaded papers
    :return: None
    """
    os.makedirs(download_dir, exist_ok=True)
    async with httpx.AsyncClient() as client:
        async with trio.open_nursery() as nursery:
            for paper in papers:
                nursery.start_soon(_download_paper, client, paper, download_dir)


async def _download_single_paper_from_link(client: httpx.AsyncClient, link: str, download_dir: str):
    """
    Download a single paper from an ArXiv link, converting it to a PDF link if necessary.

    :param client: httpx AsyncClient instance
    :param link: ArXiv link (can be PDF or abstract link)
    :param download_dir: Directory to save downloaded papers
    :return: None
    """

    # convert abstract link to PDF if necessary
    if '/abs/' in link:
        pdf_link = link.replace('/abs/', '/pdf/')
    else:
        pdf_link = link

    filename = os.path.basename(pdf_link) + '.pdf'
    filepath = os.path.join(download_dir, filename)

    try:
        response = await client.get(pdf_link)
        response.raise_for_status()
        with open(filepath, 'wb') as file:
            file.write(response.content)
        typer.echo(f"Downloaded {filename} to {filepath}")
    except httpx.HTTPStatusError as err:
        typer.echo(f"Failed to download '{pdf_link}': HTTP {err.response.status_code}")


async def download_from_links(links: List[str], download_dir: str):
    """
    Download papers from a list of ArXiv links, handling both PDF and abstract links.

    :param links: List of ArXiv links (can be PDF or abstract links)
    :param download_dir: Directory to save downloaded papers
    :return: None
    """
    os.makedirs(download_dir, exist_ok=True)
    async with httpx.AsyncClient() as client:
        async with trio.open_nursery() as nursery:
            for link in links:
                nursery.start_soon(_download_single_paper_from_link, client, link, download_dir)