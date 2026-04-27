from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db.database import init_db
from app.api import auth, chat, mood, alerts

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(mood.router)
app.include_router(alerts.router)