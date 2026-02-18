from typing import List, Dict

from arxiv_retriever.fetcher import fetch_papers, search_paper_by_title, parse_arxiv_response

import pytest


def test_parse_arxiv_response():
    """Test parse_arxiv_response function."""
    # mock xml data gotten & modified from section 4.1 of https://info.arxiv.org/help/api/user-manual.html
    mock_xml = """
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <link xmlns="http://www.w3.org/2005/Atom" href="http://arxiv.org/api/query?search_query=all:electron&amp;id_list=&amp;start=0&amp;max_results=1" rel="self" type="application/atom+xml"/>
      <title xmlns="http://www.w3.org/2005/Atom">ArXiv Query: search_query=all:electron&amp;id_list=&amp;start=0&amp;max_results=1</title>
      <updated xmlns="http://www.w3.org/2005/Atom">2007-10-08T00:00:00-04:00</updated>
      <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1000</opensearch:totalResults>
      <entry xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
        <id xmlns="http://www.w3.org/2005/Atom">http://arxiv.org/abs/hep-ex/0307015</id>
        <published xmlns="http://www.w3.org/2005/Atom">2003-07-07T13:46:39-04:00</published>
        <title xmlns="http://www.w3.org/2005/Atom">Multi-Electron Production at High Transverse Momenta in ep Collisions at HERA</title>
        <summary xmlns="http://www.w3.org/2005/Atom">Test Summary</summary>
        <author xmlns="http://www.w3.org/2005/Atom">
          <name xmlns="http://www.w3.org/2005/Atom">H1 Collaboration</name>
        </author>
        <link href="http://arxiv.org/pdf/hep-ex/0307015" rel="related" title="pdf" type="application/pdf"/>
      </entry>
    </feed>
    """

    expected_result_features = ['title', 'authors', 'summary', 'published', 'abstract_link', 'pdf_link']
    result = parse_arxiv_response(mock_xml)
    assert len(result) == 1
    assert isinstance(result, List) and isinstance(result[0], Dict)  # returns expected type
    assert all(feature in result[0] for feature in expected_result_features)
    assert result[0]['title'] == 'Multi-Electron Production at High Transverse Momenta in ep Collisions at HERA'
    assert result[0]['authors'] == ['H1 Collaboration']
    assert result[0]['summary'] == 'Test Summary'
    assert result[0]['published'] == '2003-07-07T13:46:39-04:00'
    assert result[0]['abstract_link'] == 'http://arxiv.org/abs/hep-ex/0307015'
    assert result[0]['pdf_link'] == 'http://arxiv.org/pdf/hep-ex/0307015'


# NB: following tests with authors are a bit tricky because when search is refined by author, results may be less
# than limit and should still pass. But here, I'm enforcing that the number of results returned should equal
# specified limit. This works, but there is probably better way to handle this
@pytest.mark.asyncio
async def test_fetch_papers_success(mocker):
    """Test fetch_papers function without authors"""
    mock_rate_limited_get = mocker.AsyncMock()
    mock_parse_arxiv_response = mocker.Mock()
    mocker.patch('arxiv_retriever.fetcher.rate_limited_get', mock_rate_limited_get)
    mocker.patch('arxiv_retriever.fetcher.parse_arxiv_response', mock_parse_arxiv_response)

    # Arrange
    categories = ['cs.AI', 'math.CO']
    limit = 5
    authors = None
    expected_url = ("https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:math.CO&sortBy=submittedDate"
                    "&sortOrder=descending&start=0&max_results=100")

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = '<fake_xml>Fake ArXiv response</fake_xml>'
    mock_rate_limited_get.return_value = mock_response

    expected_parsed_result = [{'title': 'Test paper', 'authors': ['John Doe'], 'pdf_link': 'http://arxiv.org/pdf/2407.0002'}] * 5
    mock_parse_arxiv_response.return_value = expected_parsed_result

    # Act
    result = await fetch_papers(categories, limit, authors)

    # Assert
    mock_rate_limited_get.assert_awaited_once_with(mocker.ANY, expected_url)
    mock_parse_arxiv_response.assert_called_once_with(mock_response.text)
    assert result == expected_parsed_result[:limit]


@pytest.mark.asyncio
async def test_fetch_papers_with_authors(mocker):
    """Test fetch_papers function with authors."""
    mock_rate_limited_get = mocker.AsyncMock()
    mock_parse_arxiv_response = mocker.Mock()
    mocker.patch('arxiv_retriever.fetcher.rate_limited_get', mock_rate_limited_get)
    mocker.patch('arxiv_retriever.fetcher.parse_arxiv_response', mock_parse_arxiv_response)

    # Arrange
    categories = ['cs.AI']
    limit = 5
    authors = ['John Doe', 'Jane Smith']
    author_logic = 'AND'
    expected_url = ('https://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+('
                    'au:"John+Doe"+AND+au:"Jane+Smith")&sortBy=submittedDate&sortOrder=descending&start=0&max_results'
                    '=100')
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = '<fake_xml>Fake ArXiv response</fake_xml>'
    mock_rate_limited_get.return_value = mock_response

    expected_parsed_result = [{'title': 'Test paper', 'authors': ['John Doe', 'Jane Smith'], 'pdf_link': 'http://arxiv.org/pdf/2407.0002'}] * 5
    mock_parse_arxiv_response.return_value = expected_parsed_result

    # Act
    result = await fetch_papers(categories, limit, authors, author_logic)

    # Assert
    mock_rate_limited_get.assert_awaited_once_with(mocker.ANY, expected_url)
    mock_parse_arxiv_response.assert_called_once_with(mock_response.text)
    assert result == expected_parsed_result[:limit]


@pytest.mark.asyncio
async def test_search_paper_by_title_success(mocker):
    """Test search_paper_by_title function without authors"""
    mock_rate_limited_get = mocker.AsyncMock()
    mock_parse_arxiv_response = mocker.Mock()
    mocker.patch('arxiv_retriever.fetcher.rate_limited_get', mock_rate_limited_get)
    mocker.patch('arxiv_retriever.fetcher.parse_arxiv_response', mock_parse_arxiv_response)

    # Arrange
    title = "Attention Is All You Need"
    limit = 5
    expected_url = (f'https://export.arxiv.org/api/query?search_query=ti:"Attention+Is+All+You+Need"&sortBy=relevance'
                    f'&sortOrder=descending&start=0&max_results=100')

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = '<fake_xml>Fake ArXiv response</fake_xml>'
    mock_rate_limited_get.return_value = mock_response

    expected_parsed_result = [{'title': 'Attention Is All You Need', 'authors': ['John Doe'], 'pdf_link': 'http://arxiv.org/pdf/2407.0003'}] * 5
    mock_parse_arxiv_response.return_value = expected_parsed_result

    # Act
    result = await search_paper_by_title(title, limit)

    # Assert
    mock_rate_limited_get.assert_awaited_once_with(mocker.ANY, expected_url)
    mock_parse_arxiv_response.assert_called_once_with(mock_response.text)
    assert result == expected_parsed_result[:limit]


@pytest.mark.asyncio
async def test_search_paper_by_title_with_authors(mocker):
    """Test search_paper_by_title function with authors."""
    mock_rate_limited_get = mocker.AsyncMock()
    mock_parse_arxiv_response = mocker.Mock()
    mocker.patch('arxiv_retriever.fetcher.rate_limited_get', mock_rate_limited_get)
    mocker.patch('arxiv_retriever.fetcher.parse_arxiv_response', mock_parse_arxiv_response)

    # Arrange
    title = "Attention Is All You Need"
    limit = 5
    authors = ['Vaswani', 'Shazeer']
    author_logic = 'AND'
    expected_url = (f'https://export.arxiv.org/api/query?search_query=ti:"Attention+Is+All+You+Need"+AND+('
                    f'au:"Vaswani"+AND+au:"Shazeer")&sortBy=relevance&sortOrder=descending&start=0&max_results=100')

    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.text = '<fake_xml>Fake ArXiv response</fake_xml>'
    mock_rate_limited_get.return_value = mock_response

    expected_parsed_result = [{'title': 'Attention Is All You Need', 'authors': ['Vaswani', 'Shazeer'], 'pdf_link': 'http://arxiv.org/pdf/2407.0004'}] * 5
    mock_parse_arxiv_response.return_value = expected_parsed_result

    # Act
    result = await search_paper_by_title(title, limit, authors, author_logic)

    # Assert
    mock_rate_limited_get.assert_awaited_once_with(mocker.ANY, expected_url)
    mock_parse_arxiv_response.assert_called_once_with(mock_response.text)
    assert result == expected_parsed_result[:limit]
