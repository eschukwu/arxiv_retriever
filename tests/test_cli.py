from itertools import combinations
from collections import namedtuple
import pytest
from typer.testing import CliRunner
from arxiv_retriever.cli import app


@pytest.fixture
def runner():
    return CliRunner()


@pytest.mark.asyncio
async def test_fetch_command_success(runner, mocker):
    mock_fetch = mocker.AsyncMock()
    mock_fetch.return_value = [
        {
            'title': 'Test Paper',
            'authors': ['John Doe'],
            'published': '2024-07-05T12:00:00Z',
            'abstract_link': 'http://arxiv.org/abs/2407.0001',
            'pdf_link': 'http://arxiv.org/pdf/2407.0001',
            'summary': 'Test paper summary.'
        }
    ]
    mocker.patch('arxiv_retriever.cli.fetch_papers', mock_fetch)

    mock_process = mocker.AsyncMock()
    mocker.patch('arxiv_retriever.cli.process_papers', mock_process)

    result = runner.invoke(app, ["fetch", "cs.AI", "--limit", "1", "--author", "John Doe"])

    assert result.exit_code == 0
    assert "Fetching up to" in result.output
    assert "1" in result.output
    assert "cs.AI" in result.output
    assert "John Doe" in result.output
    assert "OR" in result.output
    mock_fetch.assert_called_once_with(["cs.AI"], 1, ["John Doe"], "OR")
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_search_command_success(runner, mocker):
    mock_search = mocker.AsyncMock()
    mock_search.return_value = [
        {
            'title': 'Search Paper Title',
            'authors': ['John Doe'],
            'published': '2024-07-05T12:00:00Z',
            'abstract_link': 'http://arxiv.org/abs/2407.0002',
            'pdf_link': 'http://arxiv.org/pdf/2407.0002',
            'summary': 'Search paper summary.'
        }
    ]
    mocker.patch('arxiv_retriever.cli.search_paper_by_title', mock_search)
    mock_process = mocker.AsyncMock()
    mocker.patch('arxiv_retriever.cli.process_papers', mock_process)

    result = runner.invoke(app, ["search", "search paper title", "--limit", "1", "--author", "John Doe"])

    assert result.exit_code == 0
    # to create a more robust test based on output result, I will:
    # 1. split the test search query into a list
    # 2. create another list of all the possible combinations of the search query
    # 3. assert that at least one of the returned combination is in the returned result
    # I decided to take this route due to the results I was getting when
    # testing the search cli command with the query "Attention is all you need". Only one of the results returned the
    # accurate title, and it wasn't the first one. But all the results had some form of the query in them. So I basically
    # need to test that some combination of the user's search query (title) is in the title returned by the retriever
    # There's probably a better way to do this, though
    search_title_list = "search paper title".split()
    title_combinations = [' '.join(token) for i in range(len(search_title_list), 0, -1) for token in
                          combinations(search_title_list, i)]  # initial implementation was from beginning; realized
    # searching for combinations in reverse is best case
    assert any(comb.lower() in result.output.lower() for comb in title_combinations)  # main thing to check for
    assert "Searching for papers matching" in result.output
    assert "search paper title" in result.output
    assert "John Doe" in result.output
    assert "OR" in result.output
    mock_search.assert_called_once_with("search paper title", 1, ["John Doe"], "OR")
    mock_process.assert_called_once()


@pytest.mark.asyncio
async def test_download_command_success(runner, mocker):
    mock_download_from_links = mocker.AsyncMock()
    mocker.patch('arxiv_retriever.cli.download_from_links', mock_download_from_links)

    result = runner.invoke(app, ["download", "http://arxiv.org/abs/2407.0001", "--download-dir", "./test_downloads"])

    assert result.exit_code == 0
    mock_download_from_links.assert_awaited_once_with(["http://arxiv.org/abs/2407.0001"], "./test_downloads")
    assert "Download complete" in result.output
    assert "./test_downloads" in result.output


def test_version_command(runner, mocker):
    mocker.patch('arxiv_retriever.cli.vsn', return_value="1.0.0")

    # Mock sys.version_info
    VersionInfo = namedtuple('version_info', 'major minor micro releaselevel serial')
    mock_version_info = VersionInfo(major=3, minor=12, micro=0, releaselevel='final', serial=0)
    mocker.patch('arxiv_retriever.cli.sys.version_info', mock_version_info)

    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "arxiv_retriever" in result.output
    assert "1.0.0" in result.output
    assert "Python" in result.output
    assert "3.12" in result.output
    assert "Typer" in result.output
    assert "Httpx" in result.output
    assert "Trio" in result.output
