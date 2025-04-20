from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

class ChunkMetadata(BaseModel):
    text: str
    document_name: str
    vector_id: str

class Document(BaseModel):
    _id: Optional[uuid.UUID] = None
    doc_Id: str
    document_name: str
    summary: Optional[str]
    chunk_count: int
    created_at: datetime
    source: str
    chunks: Optional[List[ChunkMetadata]] = None
    
    class Config:
        arbitrary_types_allowed = True