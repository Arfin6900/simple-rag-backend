from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from bson import ObjectId

class Document(BaseModel):
    _id: Optional[ObjectId]  # This will be set to None by default
    doc_Id: str
    document_name: str
    summary: Optional[str]
    chunk_count: int
    created_at: datetime
    source: str
    
class Config:
        arbitrary_types_allowed = True