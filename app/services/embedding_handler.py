from sentence_transformers import SentenceTransformer
from app.config import pc
from pinecone import ServerlessSpec
import re
model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = 'docker-dark'


def init_index():
    if index_name not in pc.list_indexes():
        pc.create_index(
            index_name,
            dimension=384,
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(index_name)


def split_text_into_chunks(text: str, chunk_size: int = 300, overlap: int = 20):
    """Split text into word chunks with optional overlap"""
    words = re.findall(r'\w+|\W+', text)  # preserve punctuation spacing
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ''.join(words[i:i + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def embed_and_store_text(text: str):
    index = init_index()

    # Split into word-based chunks
    chunks = split_text_into_chunks(text)

    embeddings = model.encode(chunks)
    to_upsert = [
        (f'chunk-{i}', emb.tolist(), {'text': chunk})
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    index.upsert(vectors=to_upsert)
    return len(to_upsert)


def query_embeddings(query: str, top_k: int = 3):
    index = pc.Index(index_name)
    query_vector = model.encode([query])[0].tolist()
    result = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return [
        {"score": match['score'], "text": match['metadata']['text']}
        for match in result['matches']
    ]
