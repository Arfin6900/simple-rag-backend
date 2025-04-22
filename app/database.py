from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URL = os.getenv("MONGO_URL")

client = AsyncIOMotorClient(MONGO_URL)
db = client["rag_system"]

# Collections
documents_collection = db["documents"]
queries_collection = db["queries"]
chat_rooms_collection = db["chat_rooms"]
chat_messages_collection = db["chat_messages"]
