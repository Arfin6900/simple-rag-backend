# app/main.py
from fastapi import FastAPI
from app.routers import items, users, embeddings
from fastapi.middleware.cors import CORSMiddleware

index_name = "quickstart"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend's origin
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(items.router)
app.include_router(users.router)
app.include_router(embeddings.router)
