# app/services/chatbot.py

import logging
import os
import sys

from sqlalchemy import insert, select

# Core tables
from backend.app.db.models import alerts, chat_sessions, messages
from backend.app.services.emotion import detect_emotion
from backend.app.services.prompt_builder import build_prompt
from backend.app.services.risk import detect_risk

logger = logging.getLogger(__name__)

USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"

local_chatbot_instance = None


# --------------------------------------------------
# GPT FALLBACK
# --------------------------------------------------

async def get_gpt_response(prompt: str) -> str:
    return "I'm here for you. Tell me more."


# --------------------------------------------------
# LOAD LOCAL MODEL ONCE
# --------------------------------------------------

def load_local_model():
    global local_chatbot_instance

    if local_chatbot_instance:
        return local_chatbot_instance

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../")
    )

    if project_root not in sys.path:
        sys.path.append(project_root)

    from AIModels.chatbot import MistralChatbot

    bot = MistralChatbot()
    bot.load_model()

    local_chatbot_instance = bot
    return bot


# --------------------------------------------------
# GENERATE REPLY
# --------------------------------------------------

async def generate_reply(prompt: str):

    try:
        if USE_LOCAL_MODEL:
            model = load_local_model()
            return model.generate_response(prompt)

        return await get_gpt_response(prompt)

    except Exception:
        logger.exception("Reply generation failed")
        return "I'm here for you. Tell me more."


# --------------------------------------------------
# GET OR CREATE SESSION
# --------------------------------------------------

def get_or_create_session(user_id: int, db, first_message=None):

    result = db.execute(
        select(chat_sessions)
        .where(chat_sessions.c.user_id == user_id)
        .order_by(chat_sessions.c.id.desc())
    ).first()

    # optional: always create new session if desired
    title = first_message[:40] if first_message else "New Chat"

    inserted = db.execute(
        insert(chat_sessions)
        .values(
            user_id=user_id,
            title=title
        )
        .returning(chat_sessions.c.id)
    )

    session_id = inserted.scalar()

    return {"id": session_id, "user_id": user_id}


# --------------------------------------------------
# MAIN CHAT PIPELINE
# --------------------------------------------------

async def process_message(message: str, user_id: int, db):

    try:

        # ------------------------------------------------
        # 1 Emotion Detection
        # ------------------------------------------------
        try:
            emotion = detect_emotion(message)
        except Exception:
            emotion = "neutral"

        # ------------------------------------------------
        # 2 Risk Detection
        # ------------------------------------------------
        try:
            risk_data = detect_risk(message)
            risk = risk_data.get("type", "safe")
            risk_score = float(risk_data.get("score", 0))
        except Exception:
            risk_data = {"type": "safe", "score": 0}
            risk = "safe"
            risk_score = 0

        # ------------------------------------------------
        # 3 Mood Score Mapping
        # ------------------------------------------------
        mood_map = {
            "happy": 9,
            "joy": 9,
            "calm": 8,
            "neutral": 6,
            "anxious": 4,
            "fear": 4,
            "sad": 3,
            "depressed": 2,
            "anger": 3
        }

        mood_score = mood_map.get(emotion.lower(), 5)

        # ------------------------------------------------
        # 4 Prompt Build
        # ------------------------------------------------
        prompt = build_prompt(message, emotion, risk)

        # ------------------------------------------------
        # 5 Generate Reply
        # ------------------------------------------------
        reply = await generate_reply(prompt)

        # ------------------------------------------------
        # 6 Session
        # ------------------------------------------------
        chat_session = get_or_create_session(user_id, db)

        # ------------------------------------------------
        # 7 Save User Message
        # ------------------------------------------------
        user_msg_result = db.execute(
            insert(messages)
            .values(
                session_id=chat_session["id"],
                sender="user",
                message_text=message,
                emotion_label=emotion,
                risk_level=risk
            )
            .returning(messages.c.id)
        )

        user_message_id = user_msg_result.scalar()

        # ------------------------------------------------
        # 8 Save Mood Tracking
        # ------------------------------------------------
        from app.db.models import mood_tracking

        db.execute(
            insert(mood_tracking).values(
                user_id=user_id,
                mood_score=mood_score,
                emotion_label=emotion,
                notes=message
            )
        )

        # ------------------------------------------------
        # 9 Save Alerts
        # ------------------------------------------------
        if risk in ["risk", "danger", "warning"]:

            db.execute(
                insert(alerts).values(
                    user_id=user_id,
                    message_id=user_message_id,
                    risk_level=risk,
                    trigger_text=message
                )
            )

        # ------------------------------------------------
        # 10 Save Bot Reply
        # ------------------------------------------------
        db.execute(
            insert(messages).values(
                session_id=chat_session["id"],
                sender="bot",
                message_text=reply,
                emotion_label="neutral",
                risk_level="safe"
            )
        )

        db.commit()

        return {
            "reply": reply,
            "emotion": emotion,
            "risk": risk_data,
            "mood_score": mood_score,
            "session_id": chat_session["id"]
        }

    except Exception:
        db.rollback()
        logger.exception("Chat processing failed")

        return {
            "reply": "Sorry, something went wrong.",
            "emotion": "neutral",
            "risk": {"type": "safe", "score": 0},
            "mood_score": 5
        }