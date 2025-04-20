from sentence_transformers import SentenceTransformer
from app.config import pc
from pinecone import ServerlessSpec
import re
model = SentenceTransformer('all-MiniLM-L6-v2')
index_name = 'docker-dark'


def init_index():
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


def embed_and_store_text(text: str, document_name: str):
    index = init_index()

    # Split text into chunks
    chunks = split_text_into_chunks(text)

    # Generate embeddings
    embeddings = model.encode(chunks)

    # Prepare data for upserting with document name
    to_upsert = [
        (
            f'{document_name}-chunk-{i}',  # Unique ID using document name
            emb.tolist(),
            {
                'text': chunk,
                'document_name': document_name  # Metadata for filtering
            }
        )
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    # Upsert to Pinecone index
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

def get_chunks_by_document_name(document_name: str):
    index = pc.Index(index_name)
    # Query the index for all chunks associated with the document name
    response = index.query(
        top_k=1000,
        filter={"document_name": document_name},
        include_metadata=True
    )
    return [match["metadata"] for match in response["matches"]]
