
from litellm import completion
import os


openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Interact with OpenAI's GPT-4
def summarize_text(content: str, model: str = "gpt-4") -> str:
    api_key = openai_api_key if model.startswith("gpt") else gemini_api_key
    intro = ' '.join(content.split()[:300])
    response = completion(
        model=model,
        messages=[{"role": "user", "content": f"Please provide a concise summary of the following content:\n\n{intro}"}],
        api_key=api_key,
        temperature=0.5,
        max_tokens=300
    )
    return response['choices'][0]['message']['content']
    
    # Interact with OpenAI's GPT-4
def query_response_by_content(content, model: str = "gpt-4") -> str:
    api_key = openai_api_key if model.startswith("gpt") else gemini_api_key
    response = completion(
        model=model,
        messages=[{"role": "user", "content": f"you are a helpful assistant, please answer the following question based on the content provided:\n\n{content}"}],
        api_key=api_key,
        temperature=0.5,
        max_tokens=5000
    )
    return response['choices'][0]['message']['content']