from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api import alerts, auth, chat, mood
from backend.app.db.database import init_db
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(chat_router)

origins = [
    "http://localhost:3000",   # <-- Add this to match your current Vite port!
    "http://127.0.0.1:3000",   # <-- Add this too just in case
    "http://localhost:5173",   # Keep these as backups
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
@app.get("/")
def root():
    return {"message": "AI Mental Health API is running"}

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(mood.router)
app.include_router(alerts.router)

