from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from app.services.embedding_handler import embed_and_store_text, query_embeddings
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.embedding_handler import get_chunks_by_document_name, delete_document_from_pinecone
from app.database import documents_collection
from datetime import datetime
from app.llm_handler.llm_handler import summarize_text, query_response_by_content, get_relevancy_scores
from app.models.document import Document
from typing import List
import uuid


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/doc/embeddings/")
async def upload_embeddings_from_text_or_pdf(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if file:
            pdf_bytes = await file.read()
            content = extract_text_from_pdf(pdf_bytes)
            document_name = file.filename.rsplit('.', 1)[0]  # Remove extension
        elif text:
            content = text
            document_name = "manual_input"  # Or some other identifier
        else:
            raise HTTPException(status_code=400, detail="Provide text or PDF file.")

        total = embed_and_store_text(content, document_name=document_name)
        summary = summarize_text(content, model="gemini/gemini-1.5-flash")
        doc_id = str(uuid.uuid4())
        await documents_collection.insert_one({
            "_id": doc_id,
            "doc_Id": doc_id,
            "document_name": document_name,
            "summary": summary,
            "chunk_count": total,
            "created_at": datetime.now(),
            "source": "file" if file else "text"
        })
        return {"message": f"{total} embeddings uploaded for document '{document_name}'.", "content": ' '.join(content.split()[:300])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/doc/embeddings/query/")
async def query_doc_embeddings(request: QueryRequest):
    try:
        results = query_embeddings(request.query, request.top_k)
        
        # Get relevancy scores from LLM first
        relevancy_scores = get_relevancy_scores(request.query, results)
        
        # Create a map of document names to their relevancy scores
        score_map = {score["document_name"]: score["relevancy_score"] for score in relevancy_scores}
        
        # Filter results with non-zero relevancy scores
        relevant_results = []
        for result in results:
            relevancy_score = score_map.get(result["document_name"], 0)
            if relevancy_score > 0:
                relevant_results.append({
                    **result,
                    "relevancy_score": relevancy_score
                })
        
        if not relevant_results:
            return {
                "query": request.query,
                "results": "No relevant documents found only ask questions about the document",
                "embeddings": [],
                "sources": []
            }
            
        # Generate content only for relevant results
        content = query_response_by_content({"query": request.query, "results": relevant_results})
        
        # Process sources with relevancy scores
        sources = []
        for result in relevant_results:
            sources.append({
                "docName": result["document_name"],
                "relevancy": {
                    "score": result["relevancy_score"],
                    "embedding_score": round(result["score"] * 100, 2)
                },
                "content": result["text"],
                "vector_id": result["vector_id"]
            })
        
        return {
            "query": request.query, 
            "results": content, 
            "embeddings": relevant_results,
            "sources": sources
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#Get list of all documents in the database
@router.get("/doc/list/", response_model=List[Document])
async def get_all_documents():
    documents_cursor = documents_collection.find()
    documents = []
    async for document in documents_cursor:
        document["_id"] = str(document["_id"])
        document["chunks"] = None  # Set chunks to None for list view
        documents.append(Document(**document))
    return documents


#Get singleDoc
@router.get("/doc/document/{id}/", response_model=Document)
async def get_single_document_with_chunks(id: str):
    document = await documents_collection.find_one({"_id": id})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    document["_id"] = str(document["_id"])
    chunks = get_chunks_by_document_name(document["document_name"])
    document["chunks"] = chunks
    return Document(**document)

@router.delete("/doc/document/{document_name}/")
async def delete_document(document_name: str):
    try:
        # First delete from MongoDB
        result = await documents_collection.delete_one({"document_name": document_name})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found in MongoDB")
        
        # Then delete from Pinecone
        pinecone_success = delete_document_from_pinecone(document_name)
        if not pinecone_success:
            raise HTTPException(status_code=500, detail=f"Failed to delete document '{document_name}' from Pinecone")
        
        return {
            "message": f"Document '{document_name}' successfully deleted from both MongoDB and Pinecone",
            "mongo_deleted": True,
            "pinecone_deleted": True
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
