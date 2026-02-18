"""PDF text extraction utilities."""

import os
from pathlib import Path
from typing import List

from pypdf import PdfReader

from arxiv_retriever.summary_util.exceptions import LLMProviderError


# Maximum characters to extract from a PDF to avoid overwhelming LLMs
MAX_TEXT_LENGTH = 15000


class PDFExtractionError(LLMProviderError):
    """Raised when PDF text extraction fails."""

    def __init__(self, file_path: str, details: str = ""):
        self.file_path = file_path
        message = f"Failed to extract text from '{file_path}'."
        if details:
            message += f" {details}"
        super().__init__(message)


def extract_text_from_pdf(file_path: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file
        max_length: Maximum number of characters to extract

    Returns:
        Extracted text content

    Raises:
        PDFExtractionError: If extraction fails
    """
    if not os.path.exists(file_path):
        raise PDFExtractionError(file_path, "File does not exist.")

    if not file_path.lower().endswith(".pdf"):
        raise PDFExtractionError(file_path, "File is not a PDF.")

    try:
        reader = PdfReader(file_path)
        text_parts = []
        total_length = 0

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
                total_length += len(page_text)
                if total_length >= max_length:
                    break

        full_text = "\n".join(text_parts)

        if not full_text.strip():
            raise PDFExtractionError(
                file_path,
                "No text could be extracted. The PDF may be image-based.",
            )

        # Truncate to max_length
        if len(full_text) > max_length:
            full_text = full_text[:max_length] + "\n\n[Text truncated...]"

        return full_text

    except PDFExtractionError:
        raise
    except Exception as e:
        raise PDFExtractionError(file_path, str(e))


def collect_pdf_files(paths: List[str]) -> List[str]:
    """
    Collect PDF file paths from a list of files and/or directories.

    Args:
        paths: List of file paths or directory paths

    Returns:
        List of resolved PDF file paths
    """
    pdf_files = []

    for path_str in paths:
        path = Path(path_str).expanduser().resolve()

        if path.is_file() and path.suffix.lower() == ".pdf":
            pdf_files.append(str(path))
        elif path.is_dir():
            # Collect all PDFs in the directory (non-recursive)
            for pdf_path in sorted(path.glob("*.pdf")):
                pdf_files.append(str(pdf_path))
        elif path.is_file():
            # Non-PDF file - skip with warning
            pass
        else:
            # Path doesn't exist - skip
            pass

    return pdf_files


def build_pdf_prompt(file_path: str, text: str) -> str:
    """
    Build a summarization prompt for PDF content.

    Args:
        file_path: Path to the PDF file (for context)
        text: Extracted text from the PDF

    Returns:
        Formatted prompt string
    """
    filename = os.path.basename(file_path)
    return f"""Paper: {filename}

{text}

Please extract and summarize the most essential information from this paper.
Focus on the main contributions, key findings, methodology, and potential impact.
Suggest future research directions that are grounded in factual and currently available research.
Limit your response to 5-8 concise bullet points."""

