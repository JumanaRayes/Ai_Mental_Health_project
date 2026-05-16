# app/services/chatbot.py

import logging
import asyncio
import httpx

from sqlalchemy import insert, select

from backend.app.db.models import (
    alerts,
    chat_sessions,
    messages,
    mood_tracking,
)

from backend.app.services.emotion import detect_emotion
from backend.app.services.risk import detect_risk
from backend.app.services.memory import get_short_term_memory

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "mental-health-bot"


# --------------------------------------------------
# EMOTION NORMALIZATION
# --------------------------------------------------

EMOTION_MAP = {
    "negative": "sad",
    "positive": "happy",
    "fearful": "anxious",
    "angry": "anger",
    "sad": "sad",
    "happy": "happy",
    "neutral": "neutral",

    "stressed": "anxious",
    "stress": "anxious",
    "overwhelmed": "anxious",
    "worried": "anxious"
}


def normalize_emotion(emotion: str) -> str:
    if not emotion:
        return "neutral"
    return EMOTION_MAP.get(emotion.lower(), "neutral")


# --------------------------------------------------
# RISK OVERRIDE (SOFT + SAFE)
# --------------------------------------------------

def override_risk(message: str, risk_data: dict, score: float):

    msg = message.lower()

    anxiety_keywords = [
        "worried", "anxious", "stress", "future",
        "overwhelmed", "can’t cope", "cant cope",
        "hopeless"
    ]

    if any(k in msg for k in anxiety_keywords):
        score = min(score + 0.2, 1.0)

    return {
        "type": risk_data.get("type", "safe"),
        "score": score
    }


# --------------------------------------------------
# NORMALIZE HISTORY (FIXED SAFETY VERSION)
# --------------------------------------------------

def normalize_history(history):

    normalized = []

    for msg in history:

        # string fallback
        if isinstance(msg, str):
            normalized.append({
                "role": "user",
                "content": msg
            })
            continue

        # dict fallback
        if isinstance(msg, dict):
            normalized.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
            continue

        # ORM object safe handling
        sender = getattr(msg, "sender", "user")
        content = getattr(msg, "message_text", "")

        if sender not in ["user", "bot"]:
            sender = "user"

        role = "assistant" if sender == "bot" else "user"

        normalized.append({
            "role": role,
            "content": content
        })

    return normalized


# --------------------------------------------------
# BUILD LLM MESSAGES
# --------------------------------------------------

def build_messages(history, current_message: str, emotion: str, risk: str):

    system_prompt = {
    "role": "system",
    "content": f"""
You are a mental health support assistant.

STRICT BEHAVIOR RULES:
- NEVER be overly positive or motivational
- Do NOT say things like "you will be fine" or "I'm sure you'll do well"
- Do NOT assume positive outcomes
- Stay emotionally realistic and grounded

STYLE:
- Calm, supportive, emotionally accurate
- Reflect stress and pressure properly
- 2–4 sentences max

USER CONTEXT:
- Emotion: {emotion}
- Risk: {risk}

RESPONSE RULE:
If the user is stressed:
→ acknowledge pressure + normalize feeling + gentle question

Example style:
"I hear that this feels stressful, especially with something as important as a graduation project. What part of it feels most overwhelming right now?"
"""
}

    messages = [system_prompt]
    messages.extend(normalize_history(history))

    messages.append({
        "role": "user",
        "content": current_message
    })

    return messages


# --------------------------------------------------
# LLM CALL (OLLAMA - STABLE)
# --------------------------------------------------

async def generate_reply(history, current_message, emotion, risk):

    payload = {
        "model": OLLAMA_MODEL,
        "messages": build_messages(history, current_message, emotion, risk),
        "stream": False,
    }

    timeout = httpx.Timeout(60.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:

        for attempt in range(3):

            try:
                response = await client.post(
                    OLLAMA_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                # 🔥 IMPORTANT: show real Ollama error if it fails
                if response.status_code != 200:
                    logger.error(
                        f"Ollama error {response.status_code}: {response.text}"
                    )
                    continue

                data = response.json()

                # safe extraction
                reply = (
                    data.get("message", {})
                        .get("content", "")
                        .strip()
                )

                if reply:
                    return reply

                return "I'm here for you. Tell me more."

            except httpx.TimeoutException:
                logger.warning(f"LLM timeout attempt {attempt + 1}")

            except httpx.RequestError as e:
                logger.error(f"LLM request error attempt {attempt + 1}: {e}")

            except Exception as e:
                logger.error(f"LLM unexpected error attempt {attempt + 1}: {e}")

    return "I'm here for you. Tell me more."


# --------------------------------------------------
# SESSION HANDLING
# --------------------------------------------------

def get_or_create_session(user_id: int, db, first_message=None):

    session = db.execute(
        select(chat_sessions)
        .where(chat_sessions.c.user_id == user_id)
        .order_by(chat_sessions.c.id.desc())
    ).first()

    if session:
        return {"id": session.id, "user_id": user_id}

    title = (first_message or "New Chat")[:40]

    new_session = db.execute(
        insert(chat_sessions)
        .values(user_id=user_id, title=title)
        .returning(chat_sessions.c.id)
    )

    return {
        "id": new_session.scalar(),
        "user_id": user_id
    }


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

async def process_message(message: str, user_id: int, db):

    try:

        # -------------------------
        # Emotion detection
        # -------------------------
        try:
            emotion = detect_emotion(message)
            emotion = normalize_emotion(emotion)
        except Exception:
            emotion = "neutral"
            
        if "stress" in message.lower() or "stressed" in message.lower():
            emotion = "anxious"

        # -------------------------
        # Risk detection
        # -------------------------
        try:
            risk_data = detect_risk(message)
            risk = risk_data.get("type", "safe")
            risk_score = float(risk_data.get("score", 0))
        except Exception:
            risk_data = {"type": "safe", "score": 0}
            risk = "safe"
            risk_score = 0

        risk_data = override_risk(message, risk_data, risk_score)
        risk = risk_data["type"]
        risk_score = float(risk_data["score"])

        # -------------------------
        # Mood scoring (FIXED)
        # -------------------------
        mood_map = {
            "happy": 9,
            "calm": 8,
            "neutral": 6,
            "anxious": 4,
            "fear": 4,
            "sad": 3,
            "depressed": 2,
            "anger": 3,
            "angry": 3,
            "stressed": 4
        }

        mood_score = mood_map.get(emotion, 5)

        # -------------------------
        # Session
        # -------------------------
        session = get_or_create_session(user_id, db, message)

        # -------------------------
        # Memory
        # -------------------------
        history = get_short_term_memory(
            session["id"],
            db,
            limit=8
        )

        # -------------------------
        # LLM response
        # -------------------------
        reply = await generate_reply(history, message, emotion, risk)

        # -------------------------
        # Save user message
        # -------------------------
        user_msg = db.execute(
            insert(messages)
            .values(
                session_id=session["id"],
                sender="user",
                message_text=message,
                emotion_label=emotion,
                risk_level=risk,
            )
            .returning(messages.c.id)
        )

        user_message_id = user_msg.scalar()

        # -------------------------
        # Mood tracking
        # -------------------------
        db.execute(
            insert(mood_tracking).values(
                user_id=user_id,
                mood_score=mood_score,
                emotion_label=emotion,
                notes=message,
            )
        )

        # -------------------------
        # Alerts
        # -------------------------
        if risk in ["high", "danger", "critical"] or risk_score > 0.7:

            db.execute(
                insert(alerts).values(
                    user_id=user_id,
                    message_id=user_message_id,
                    risk_level=risk,
                    trigger_text=message,
                )
            )

        # -------------------------
        # Save bot reply
        # -------------------------
        db.execute(
            insert(messages).values(
                session_id=session["id"],
                sender="bot",
                message_text=reply,
                emotion_label="neutral",
                risk_level="safe",
            )
        )

        db.commit()

        return {
            "reply": reply,
            "emotion": emotion,
            "risk": risk_data,
            "mood_score": mood_score,
            "session_id": session["id"],
        }

    except Exception:
        db.rollback()
        logger.exception("Chat processing failed")

        return {
            "reply": "Sorry, something went wrong.",
            "emotion": "neutral",
            "risk": {"type": "safe", "score": 0},
            "mood_score": 5,
        }
