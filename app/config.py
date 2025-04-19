from pinecone import Pinecone, ServerlessSpec
import os
pc = Pinecone(api_key=os.getenv("PINECONE_API"))
