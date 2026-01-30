import os
from openai import OpenAI

from dotenv import load_dotenv

# load environment variables
load_dotenv()

# define model to use
MODEL = "gpt-4o-mini-2024-07-18"


def get_llm_response(prompt: str) -> str:
    """Prompt GPT-4o-mini for a response"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), project="proj_X4jQQNOuVT4btKtekQhBo5y4")
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant specializing in summarizing scientific papers and "
                           "extracting the most meaningful parts of the paper as simply and concisely as possible.",
                "name": "arxiv-bot",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=MODEL,
        temperature=0,
    )
    return response.choices[0].message.content
