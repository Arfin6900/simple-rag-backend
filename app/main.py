# app/main.py
from fastapi import FastAPI
from app.routers import items, users, embeddings

index_name = "quickstart"

app = FastAPI()

app.include_router(items.router)
app.include_router(users.router)
app.include_router(embeddings.router)
