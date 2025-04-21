from litellm import completion
import os
from typing import List, Dict, Any


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

def get_relevancy_scores(query: str, documents: List[Dict[str, Any]], model: str = "gpt-4") -> List[Dict[str, Any]]:
    api_key = openai_api_key if model.startswith("gpt") else gemini_api_key
    
    # Prepare the prompt
    prompt = f"""Given the following query and documents, please rate each document's relevance to the query on a scale of 0-100.
    Only return a JSON array of objects with 'document_name' and 'relevancy_score' fields.
    Do not include any explanations or additional text.

    Query: {query}

    Documents:
    {[doc['document_name'] for doc in documents]}

    Return format example:
    [
        {{"document_name": "doc1", "relevancy_score": 85}},
        {{"document_name": "doc2", "relevancy_score": 60}}
    ]
    """
    
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key,
        temperature=0.3,
        max_tokens=500
    )
    
    # Parse the response to get the relevancy scores
    try:
        scores = eval(response['choices'][0]['message']['content'])
        return scores
    except:
        # If parsing fails, return default scores
        return [{"document_name": doc["document_name"], "relevancy_score": 50} for doc in documents]