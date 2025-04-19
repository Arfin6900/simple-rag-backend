from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from app.services.embedding_handler import embed_and_store_text, query_embeddings
from app.services.pdf_extractor import extract_text_from_pdf

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


@router.post("/embeddings/")
async def upload_embeddings_from_text_or_pdf(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if file:
            pdf_bytes = await file.read()
            content = extract_text_from_pdf(pdf_bytes)
        elif text:
            content = text
        else:
            raise HTTPException(status_code=400, detail="Provide text or PDF file.")

        total = embed_and_store_text(content)
        return {"message": f"{total} embeddings uploaded."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/embeddings/query")
async def query_doc_embeddings(request: QueryRequest):
    try:
        results = query_embeddings(request.query, request.top_k)
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
