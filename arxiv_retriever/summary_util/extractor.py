from typing import List, Dict
from arxiv_retriever.summary_util.llm_interface import get_llm_response


# Current focus is on extracting essential info from paper abstract.
# future implementation might include building a full RAG system that
# feeds the actual pdf into the model
def extract_essential_info(papers: List[Dict]) -> List[Dict]:
    """Extract essential information from papers."""
    essential_info = []
    for paper in papers:
        prompt = f"""
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Summary: {paper['summary']}
        
        
        Please extract and summarize the most essential information from this paper abstract.
        Focus on the main contributions, key findings, and potential impact of the research.
        Suggest future research directions that is grounded in factual and currently available research.
        Limit your response to 3-5 concise bullet points.
        """

        response = get_llm_response(prompt)
        essential_info.append({
            'title': paper['title'],
            'extracted_info': response
        })
    return essential_info
