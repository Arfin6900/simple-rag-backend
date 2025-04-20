from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from app.services.embedding_handler import embed_and_store_text, query_embeddings
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.embedding_handler import get_chunks_by_document_name
from app.database import documents_collection
from datetime import datetime
from app.llm_handler.llm_handler import summarize_text
from app.models.document import Document
from typing import List
from bson import ObjectId


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
        summary = summarize_text(content, model="gemini/gemini-1.5-flash")  # Get the first 300 characters for the summary
        await documents_collection.insert_one({
            "doc_Id": document_name,
            "document_name": document_name,
            "summary": summary,
            "chunk_count": total,
            "created_at": datetime.now(),
            "source": "file" if file else "text"
        })
        return {"message": f"{total} embeddings uploaded for document '{document_name}'.",content:' '.join(content.split()[:300])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/doc/embeddings/query")
async def query_doc_embeddings(request: QueryRequest):
    try:
        results = query_embeddings(request.query, request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

#Get list of all documents in the database
@router.get("/doc/list/", response_model=List[Document])
async def get_all_documents():
    documents_cursor = documents_collection.find()
    documents = []
    async for document in documents_cursor:
        documents.append(Document(**document))
    return documents


#Get singleDoc
@router.get("/doc/{id}/", response_model=Document)
async def get_single_document_with_chunks(id: str):
    document = await documents_collection.find_one({"_id": ObjectId(id)})
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    document["doc_Id"] = str(document.pop("_id"))
    chunks = get_chunks_by_document_name(document["document_name"])
    document["chunks"] = chunks
    return Document(**document)
    
