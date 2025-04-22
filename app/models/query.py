from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from bson import ObjectId
from enum import Enum

class Provider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"

class Query(BaseModel):
    id: Optional[str] = None
    query: str
    document_name: Optional[str] = None
    chat_room_id: Optional[str] = None
    timestamp: datetime
    response: Optional[str] = None
    relevancy_scores: Optional[List[dict]] = None
    user_id: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True

class ChatRoom(BaseModel):
    id: Optional[str] = None
    name: str
    created_at: datetime
    user_id: Optional[str] = None
    contexts: List[str] = []  # Array of document names
    provider: Provider = Provider.OPENAI
    is_active: bool = True
    
    class Config:
        arbitrary_types_allowed = True

class ChatMessage(BaseModel):
    id: Optional[str] = None
    chat_room_id: str
    query: str
    response: str
    timestamp: datetime
    relevancy_scores: Optional[List[dict]] = None
    
    class Config:
        arbitrary_types_allowed = True 