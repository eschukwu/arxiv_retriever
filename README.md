# ArXiv Retriever

## Status: Maintenance Mode

**Note:** This project is currently in maintenance mode. While I am not actively developing new features, I will continue
to address critical issues and security vulnerabilities as time permits. Users are welcome to fork the repository if
they wish to extend its functionality. Please refer to [Maintenance Policy](#maintenance-policy) for more information.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Environment Setup](#environment-setup)
- [Installation](#installation)
- [Usage](#usage)
- [LLM Providers](#llm-providers)
- [Contributing](#contributing)
- [Maintenance Policy](#maintenance-policy)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Introduction
`arxiv_retriever` is a lightweight command-line tool designed to automate the retrieval, downloading, and
summarization of research papers from [ArXiv](https://arxiv.org/). The retrieval can be done using specified ArXiv
categories, full or partial titles of papers, or links to the papers. Paper retrieval can be refined by author.

Papers can be summarized using multiple LLM providers — **Ollama** (local, default), **Claude** (Anthropic), or
**Gemini** (Google) — directly from the terminal.

**NOTE:** My tests indicate that when searching for a really long title, using the partial title and then refining by author
yields better results, as opposed to searching with the full title or even searching with the full title and refining by
author. However, the tests are not exhaustive.

This tool is built using Python and leverages the Typer library for the command-line interface, Rich for enhanced
terminal output, and the Python ElementTree XML package for parsing XML responses from the arXiv API. It can be useful
for researchers, engineers, or students who want to quickly retrieve an ArXiv paper or keep abreast of latest research
in their field without leaving their terminal/workstation.

Although my current focus while building `arxiv_retriever` is the computer science archive, it can be easily
used with categories from other areas on arxiv, e.g., `math.CO`.

## Features
- Fetch the most recent papers from specified ArXiv categories
- Search for papers on ArXiv using full or partial title
- Refine fetch and search by author(s) for more precise results
- Specify logic for combination of multiple authors ('AND' or 'OR') during retrieval
- Download papers after they are retrieved
- **Summarize PDF papers** using LLM providers (Ollama, Claude, Gemini)
  - Batch summarization of multiple papers at once
  - Save summaries to JSON files
- View paper details including title, authors, abstract, publication date, and links
- **Rich terminal display** with styled panels, Markdown rendering, and color-coded output
- Multi-provider LLM support with shorthand syntax (e.g., `--model claude`)
- Configurable number of results to fetch
- Easy-to-use command-line interface built with Typer

## Environment Setup

Environment variables are used to configure LLM providers for the paper summarization feature. **Ollama** is the
default provider and requires no API keys (it runs locally).

### Environment Variables

| Variable | Provider | Required | Default |
|----------|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Claude | Yes (for Claude) | — |
| `GEMINI_API_KEY` | Gemini | Yes (for Gemini) | — |
| `OLLAMA_BASE_URL` | Ollama | No | `http://localhost:11434` |
| `ARXIV_RETRIEVER_DEFAULT_MODEL` | All | No | `ollama:llama3` |

### Setting Environment Variables

#### On Unix-like systems (Linux, macOS)
In your terminal, run:
```shell
export ANTHROPIC_API_KEY=<your-anthropic-key>
export GEMINI_API_KEY=<your-gemini-key>
```
To ensure this works across all shell instances, add the above lines to your shell configuration file
(e.g., `~/.bashrc`, `~/.zshrc`, or `~/.profile`).

#### On Windows
1. Open the Start menu and search for "Environment Variables"
2. Click on the "Edit system environment variables" option.
3. In the System Properties window, click on the "Environment Variables" button
4. Under "User variables", click "New"
5. Set the variable name and value for each key.

**NOTE:** Keep your API keys confidential and do not share them publicly.

## Installation

### Install  from PyPI (Recommended):

```shell
pip install --upgrade arxiv-retriever
```

### Install from Source Distribution

If you need a specific version or want to install from a source distribution:

1. Download the source distribution (.tar.gz file) from PyPI or the GitHub releases page.

2. Install using pip:
   ```bash
   pip install axiv-x.y.z.tar.gz
   ```
   Replace `x.y.z` with the version number.

This method can be useful if you need a specific version or are in an environment without direct access to PyPI.

### Install for Development and Testing

To install the latest development version from source:
1. Ensure you have [uv](https://docs.astral.sh/uv/) installed.
2. Clone the repository:
    ```shell
    git clone https://github.com/MimicTester1307/arxiv_retriever.git
    cd arxiv_retriever
    ```
3. Install the project and its dependencies:
    ```shell
    uv sync
    ```
4. Run tests to ensure everything is set up correctly:
    ```shell
    uv run pytest
    ```
5. Run the CLI:
    ```shell
    uv run axiv --help
    ```

## Usage

After installation, use the package via the `axiv` command. To view available commands: `axiv --help` or `axiv`

### Note on Package and Command Names

- **Package Name**: The package is named `arxiv_retriever`. This is the name you use when installing via pip or referring to the project.
- **Command Name**: After installation, you interact with the tool using the `axiv` command in your terminal.

This distinction allows for a more concise command while maintaining a descriptive package name.

### Basic Commands

- `fetch`: Fetch papers from ArXiv based on categories, refined by options.
- `search`: Search for papers on ArXiv using title, refined by options.
- `download`: Download papers from ArXiv using their links (PDF or abstract links).
- `summarize`: Summarize one or more PDF papers using an LLM provider.
- `version`: Display version information for arxiv_retriever and core dependencies.

### Sample Usage

#### Fetch
To retrieve the most recent computer science papers by categories, use the `fetch` command followed by the categories and 
options:
```shell
axiv fetch [OPTIONS] CATEGORIES...
```

#### Search
To search for a paper by title, use the `search` command followed by the title and options:
```shell 
axiv search [OPTIONS] TITLE
```

#### CLI Options
Due to how most CLI frameworks (including Typer) handle arguments vs options, if you want to specify multiple options (in this case, authors)
to refine your `search` or `fetch` command by, you will have to call the option multiple times. That is,
`--author <author> --author <author>` as opposed to `--author <author> <author>`. Alternatively, you can use `-a` rather
than `--author`

#### Downloading your research papers
There are multiple ways to download your research paper using `axiv`:
- use `axiv download [OPTIONS] LINKS...` to download the paper directly from the link
- confirm if you want to download the retrieved papers using `fetch` or `search` when asked by the CLI

With option 1, the file is named using the URL's basename, e.g. `2407.09298v1.pdf`.

With options 2, the file is named using the title retrieved from the XML data when parsing.

**NOTE:** If the file name exists, it is overwritten.


### Examples
Fetch the latest 5 papers in the cs.AI OR cs.GL categories:
```shell
axiv fetch cs.AI cs.GL --limit 5
```
*Outputs `limit` papers sorted by `submittedDate` in descending order, filtered by `authors`*

Refine fetch using multiple authors
```shell
axiv fetch cs.AI -a omar -a matei
```

Add logic for creating query when multiple authors are supplied using `--author-logic` or `-l`:
```shell
axiv fetch cs.AI math.CO -a "John Doe" -a "Jane Smith" --author-logic AND
```

Fetch papers matching the title, "Attention is all you need", refined by author "Ashish":
```shell
axiv search "Attention is all you need" --limit 5 --author "Ashish"
```

Download papers using links:

- download using link to abstract:
    ```shell
    axiv download https://arxiv.org/abs/2407.20214v1
    ```
- download using link to pdf:
    ```shell
    axiv download https://arxiv.org/pdf/2407.20214v1
    ```

#### Summarize
Summarize downloaded PDF papers using an LLM:
```shell
# Summarize a single paper (uses Ollama by default — local, no rate limits)
axiv summarize paper.pdf

# Summarize multiple papers
axiv summarize paper1.pdf paper2.pdf

# Summarize all PDFs in a directory
axiv summarize ./arxiv_downloads/

# Use a specific provider
axiv summarize paper.pdf --model claude
axiv summarize paper.pdf --model gemini

# Use a specific model
axiv summarize paper.pdf --model claude:claude-sonnet-4-6

# Save summaries to JSON
axiv summarize ./arxiv_downloads/ --save
```

## LLM Providers

`arxiv_retriever` supports multiple LLM providers for paper summarization. **Ollama is the default** — it runs
locally and has no API rate limits or costs.

### Provider Format

Use `--model provider:model_name` or just `--model provider` (uses the default model for that provider):

| Provider | Default Model | Shorthand | Requires |
|----------|--------------|-----------|----------|
| Ollama | `llama3` | `--model ollama` | [Ollama](https://ollama.com/) running locally |
| Claude | `claude-sonnet-4-6` | `--model claude` | `ANTHROPIC_API_KEY` env var |
| Gemini | `gemini-3-flash-preview` | `--model gemini` | `GEMINI_API_KEY` env var |

### Examples
```shell
# Use default (Ollama)
axiv summarize paper.pdf

# Use Claude with default model
axiv summarize paper.pdf --model claude

# Use Gemini with a specific model
axiv summarize paper.pdf --model gemini:gemini-2.0-flash

# Set a custom default model via environment variable
export ARXIV_RETRIEVER_DEFAULT_MODEL=claude:claude-sonnet-4-6
axiv summarize paper.pdf
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any features, bug fixes, or
enhancements.

### Note on Testing

Currently, all 35 tests pass. Refactoring the tests for asynchrony was an interesting challenge.
Discussions and contributions regarding the asynchronous implementation are particularly welcome.

```shell
uv run pytest
```

Contact me via email or leave a comment on the [Notion project tracker](https://clover-gymnast-aeb.notion.site/ArXiv-Retriever-630d06d96edf4bfea17248cc890c021e?pvs=4).

## Maintenance Policy
This project is currently in maintenance mode. Here is what you can expect:
- Security vulnerabilities and bugs will be addressed as time permits.
- Pull requests for bug fixes will be considered. 
- Feature requested are unlikely to be implemented by the maintainer, but forks and extensions are encouraged.

For any questions, concerns, or comments, please open an issue in the GitHub repository.

## License
This project is licensed under the MIT license. See the LICENSE file for more details.

## Acknowledgements
- [Typer](https://typer.tiangolo.com/) for the command-line interface
- [Rich](https://rich.readthedocs.io/) for enhanced terminal output (panels, tables, Markdown rendering)
- [ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html) for XML parsing
- [arXiv API](https://info.arxiv.org/help/api/basics.html) for providing access to paper metadata via a well-designed API
- [Trio](https://trio.readthedocs.io/en/stable/index.html) and [HTTPx](https://www.python-httpx.org/) for the asynchronous features
- [pypdf](https://pypdf.readthedocs.io/) for PDF text extraction
- [Ollama](https://ollama.com/) for local LLM inference
- [Anthropic](https://docs.anthropic.com/) and [Google GenAI](https://ai.google.dev/) SDKs for cloud LLM providers
- [Dead Simple Python](https://www.amazon.de/-/en/Jason-C-McDonald/dp/1718500920) for helping me advance my knowledge of Python