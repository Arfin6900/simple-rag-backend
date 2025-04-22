from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List, Dict
from app.services.embedding_handler import embed_and_store_text, query_embeddings
from app.services.pdf_extractor import extract_text_from_pdf
from app.services.embedding_handler import get_chunks_by_document_name, delete_document_from_pinecone
from app.database import documents_collection, queries_collection, chat_rooms_collection, chat_messages_collection
from datetime import datetime, timedelta
from app.llm_handler.llm_handler import summarize_text, query_response_by_content, get_relevancy_scores
from app.models.document import Document
from app.models.query import Query, ChatRoom, ChatMessage
import uuid
from bson import ObjectId


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    document_name: Optional[str] = None
    chat_room_id: Optional[str] = None
    user_id: Optional[str] = None


class DocumentTypeStats(BaseModel):
    name: str
    value: int


class QueryActivityStats(BaseModel):
    name: str
    queries: int


class RecentQuery(BaseModel):
    id: str
    query: str
    timestamp: str


class DashboardStats(BaseModel):
    total_documents: int
    documents_change: int
    uploaded_today: int
    uploaded_change: int
    total_queries: int
    queries_change: int
    system_status: str
    system_status_description: str
    document_types: List[DocumentTypeStats]
    query_activity: List[QueryActivityStats]


class QueryStats(BaseModel):
    query_activity: List[QueryActivityStats]
    recent_queries: List[RecentQuery]


class ChatRoomRequest(BaseModel):
    name: str
    contexts: Optional[List[str]] = None
    provider: Optional[str] = "openai"
    user_id: Optional[str] = None


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
        # If chat_room_id is provided, get the chat room and its contexts
        contexts = None
        provider = "gemini"
        if request.chat_room_id:
            chat_room = await chat_rooms_collection.find_one({"_id": request.chat_room_id, "is_active": True})
            if not chat_room:
                raise HTTPException(status_code=404, detail="Chat room not found or inactive")
            contexts = chat_room.get("contexts", [])
            provider = chat_room.get("provider", "gemini")
        
        # If contexts are specified, only search within those documents
        if contexts:
            results = []
            for doc_name in contexts:
                doc_results = query_embeddings(request.query, request.top_k)
                filtered_results = [r for r in doc_results if r["document_name"] == doc_name]
                results.extend(filtered_results)
        else:
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
            response = {
                "query": request.query,
                "results": "No relevant documents found only ask questions about the document",
                "embeddings": [],
                "sources": []
            }
        else:
            # Generate content only for relevant results using the specified provider
            content = query_response_by_content({"query": request.query, "results": relevant_results}, model=f"{provider}/gpt-4" if provider == "openai" else "gemini/gemini-1.5-flash")
            
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
            
            response = {
                "query": request.query, 
                "results": content, 
                "embeddings": relevant_results,
                "sources": sources
            }

        # Store the query in the database
        query_id = str(uuid.uuid4())
        query_data = {
            "_id": query_id,
            "query": request.query,
            "document_name": request.document_name,
            "chat_room_id": request.chat_room_id,
            "timestamp": datetime.now(),
            "response": response["results"],
            "relevancy_scores": [{"document_name": r["document_name"], "score": r["relevancy_score"]} for r in relevant_results],
            "user_id": request.user_id
        }
        await queries_collection.insert_one(query_data)

        # If this is part of a chat room, also store it as a chat message
        if request.chat_room_id:
            chat_message = {
                "_id": str(uuid.uuid4()),
                "chat_room_id": request.chat_room_id,
                "query": request.query,
                "response": response["results"],
                "timestamp": datetime.now(),
                "relevancy_scores": [{"document_name": r["document_name"], "score": r["relevancy_score"]} for r in relevant_results]
            }
            await chat_messages_collection.insert_one(chat_message)

        return response
    except HTTPException:
        raise
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
    

@router.post("/chat/room/")
async def create_chat_room(request: ChatRoomRequest):
    try:
        # Validate provider
        if request.provider not in ["openai", "gemini"]:
            raise HTTPException(status_code=400, detail="Provider must be either 'openai' or 'gemini'")
        
        # Validate contexts if provided
        if request.contexts:
            for doc_name in request.contexts:
                doc = await documents_collection.find_one({"document_name": doc_name})
                if not doc:
                    raise HTTPException(status_code=404, detail=f"Document '{doc_name}' not found")
        
        chat_room_id = str(uuid.uuid4())
        chat_room = {
            "_id": chat_room_id,
            "name": request.name,
            "created_at": datetime.now(),
            "user_id": request.user_id,
            "contexts": request.contexts or [],
            "provider": request.provider,
            "is_active": True
        }
        await chat_rooms_collection.insert_one(chat_room)
        return {"chat_room_id": chat_room_id, "message": "Chat room created successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/room/{chat_room_id}/")
async def delete_chat_room(chat_room_id: str):
    try:
        # First check if chat room exists
        chat_room = await chat_rooms_collection.find_one({"_id": chat_room_id})
        if not chat_room:
            raise HTTPException(status_code=404, detail="Chat room not found")
        
        # Soft delete the chat room
        await chat_rooms_collection.update_one(
            {"_id": chat_room_id},
            {"$set": {"is_active": False}}
        )
        
        return {"message": "Chat room deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/room/{chat_room_id}/messages/")
async def get_chat_messages(chat_room_id: str):
    try:
        messages = []
        async for message in chat_messages_collection.find({"chat_room_id": chat_room_id}).sort("timestamp", -1):
            message["_id"] = str(message["_id"])
            messages.append(ChatMessage(**message))
        return messages
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/rooms/")
async def get_user_chat_rooms(user_id: Optional[str] = None):
    try:
        query = {"is_active": True}
        if user_id:
            query["user_id"] = user_id
            
        chat_rooms = []
        async for room in chat_rooms_collection.find(query).sort("created_at", -1):
            room["_id"] = str(room["_id"])
            chat_rooms.append(ChatRoom(**room))
        return chat_rooms
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/dashboard/", response_model=DashboardStats)
async def get_dashboard_stats():
    try:
        # Get current date and dates for comparison
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        yesterday_start = today_start - timedelta(days=1)
        last_week_start = today_start - timedelta(days=7)
        
        # Get total documents and change from last week
        total_docs = await documents_collection.count_documents({})
        last_week_docs = await documents_collection.count_documents({
            "created_at": {"$lt": last_week_start}
        })
        documents_change = total_docs - last_week_docs
        
        # Get documents uploaded today and change from yesterday
        uploaded_today = await documents_collection.count_documents({
            "created_at": {"$gte": today_start}
        })
        uploaded_yesterday = await documents_collection.count_documents({
            "created_at": {"$gte": yesterday_start, "$lt": today_start}
        })
        uploaded_change = uploaded_today - uploaded_yesterday
        
        # Get document types distribution
        doc_types = await documents_collection.aggregate([
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$project": {"name": "$_id", "value": "$count", "_id": 0}}
        ]).to_list(length=None)
        
        # Calculate percentages for document types
        total = sum(doc["value"] for doc in doc_types)
        document_types = [
            {"name": doc["name"].upper(), "value": round((doc["value"] / total) * 100)}
            for doc in doc_types
        ]
        
        # Get query activity for the last 7 days
        query_activity = []
        for i in range(7):
            day = today_start - timedelta(days=i)
            next_day = day + timedelta(days=1)
            query_count = await queries_collection.count_documents({
                "timestamp": {"$gte": day, "$lt": next_day}
            })
            query_activity.append({
                "name": day.strftime("%a"),
                "queries": query_count
            })
        query_activity.reverse()
        
        # Get total queries and change from last week
        total_queries = await queries_collection.count_documents({})
        last_week_queries = await queries_collection.count_documents({
            "timestamp": {"$lt": last_week_start}
        })
        queries_change = total_queries - last_week_queries
        
        return DashboardStats(
            total_documents=total_docs,
            documents_change=documents_change,
            uploaded_today=uploaded_today,
            uploaded_change=uploaded_change,
            total_queries=total_queries,
            queries_change=queries_change,
            system_status="Healthy",
            system_status_description="All systems operational",
            document_types=document_types,
            query_activity=query_activity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/queries/", response_model=QueryStats)
async def get_query_stats():
    try:
        # Get query activity for the last 7 days
        query_activity = []
        now = datetime.now()
        today_start = datetime(now.year, now.month, now.day)
        
        for i in range(7):
            day = today_start - timedelta(days=i)
            next_day = day + timedelta(days=1)
            query_count = await queries_collection.count_documents({
                "timestamp": {"$gte": day, "$lt": next_day}
            })
            query_activity.append({
                "name": day.strftime("%a"),
                "queries": query_count
            })
        query_activity.reverse()
        
        # Get recent queries
        recent_queries = []
        async for query in queries_collection.find().sort("timestamp", -1).limit(4):
            query["_id"] = str(query["_id"])
            recent_queries.append({
                "id": query["_id"],
                "query": query["query"],
                "timestamp": format_timestamp(query["timestamp"])
            })
        
        return QueryStats(
            query_activity=query_activity,
            recent_queries=recent_queries
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def format_timestamp(timestamp: datetime) -> str:
    now = datetime.now()
    diff = now - timestamp
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hours ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minutes ago"
    else:
        return "just now"
    
