from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from backend.app.db.database import get_db
from backend.app.db.models import chat_sessions, messages
from backend.app.schemas.chat_schema import ChatRequest
from backend.app.services.chatbot import process_message
from backend.app.utils.dependencies import get_current_user

router = APIRouter(prefix="/chat", tags=["Chat"])


# -----------------------------------------
# GET ALL USER SESSIONS
# -----------------------------------------
@router.get("/sessions")
def get_sessions(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    result = db.execute(
        select(chat_sessions)
        .where(chat_sessions.c.user_id == int(user["user_id"]))
        .order_by(chat_sessions.c.id.desc())
    ).fetchall()

    return [
        {
            "id": row._mapping["id"],
            "title": row._mapping.get("title", "New Chat"),
            "created_at": row._mapping["created_at"]
        }
        for row in result
    ]


# -----------------------------------------
# GET SESSION HISTORY
# -----------------------------------------
@router.get("/history/{session_id}")
def get_history(
    session_id: int,
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    session = db.execute(
        select(chat_sessions)
        .where(chat_sessions.c.id == session_id)
        .where(chat_sessions.c.user_id == int(user["user_id"]))
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    result = db.execute(
        select(messages)
        .where(messages.c.session_id == session_id)
        .order_by(messages.c.id.asc())
    ).fetchall()

    return [
        {
            "sender": row._mapping["sender"],
            "message": row._mapping["message_text"],
            "emotion": row._mapping["emotion_label"],
            "risk": row._mapping["risk_level"],
            "created_at": row._mapping["created_at"]
        }
        for row in result
    ]


# -----------------------------------------
# ✅ CHATBOT ENDPOINT (FIXED)
# -----------------------------------------
@router.post("/message")
async def send_message(
    data: ChatRequest,
    user=Depends(get_current_user),
    db=Depends(get_db)
):

    result = await process_message(
        message=data.message,
        user_id=int(user["user_id"]),
        db=db
    )

    return result