from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.app.api import alerts, auth, chat, mood
from backend.app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)
 
 
@app.get("/")
def root():
    return {"message": "AI Mental Health API is running"}



app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(mood.router)
app.include_router(alerts.router)