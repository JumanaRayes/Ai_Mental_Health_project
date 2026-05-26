# app/services/chatbot.py

import logging
import asyncio
import httpx
import re 

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
OLLAMA_MODEL = "mistral-local"


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
# REGEX HELPER FOR STRICT MATCHING
# --------------------------------------------------

def contains_whole_word(keywords: list, text: str) -> bool:
    """
    Ensures substrings do not cause false positives.
    e.g., 'end' will no longer match inside 'excited'.
    """
    if not keywords:
        return False
    pattern = r'\b(' + '|'.join(map(re.escape, keywords)) + r')\b'
    return bool(re.search(pattern, text))


# --------------------------------------------------
# RISK OVERRIDE (HARD GUARDRAILS FOR CRISIS)
# --------------------------------------------------

def override_risk(message: str, risk_data: dict, score: float):
    msg = message.lower()

    # 🚨 CRITICAL: Explicit crisis strings
    crisis_keywords = [
        "end everything", "hurt myself", "commit suicide", "kill myself", 
        "ending my life", "want to die", "bottle of pills", "ending seems like",
        "suicidal", "self-harm", "end it all", "end this"
    ]

    anxiety_keywords = [
        "worried", "anxious", "stress", "future",
        "overwhelmed", "can’t cope", "cant cope",
        "hopeless"
    ]

    # Use regex safety match to force high risk status
    if contains_whole_word(crisis_keywords, msg):
        return {
            "type": "high",
            "score": 1.0
        }

    # Soft bump for anxiety indicators
    if contains_whole_word(anxiety_keywords, msg):
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


# BUILD LLM MESSAGES

def build_messages(history, current_message: str, emotion: str, risk: str):

    system_prompt = {
    "role": "system",
    "content": f"""
    You are a mental health support assistant.

    STRICT RULES:
    - NEVER sound robotic
    - NEVER ignore the user's emotional context
    - NEVER give generic therapist questions
    - NEVER suddenly change the topic
    - NEVER be overly optimistic

    RESPONSE STYLE:
    - Emotionally validating
    - Natural and human
    - Short (2-4 sentences)
    - Supportive but realistic

    IMPORTANT:
    If the user expresses:
    - exhaustion
    - exam stress
    - burnout
    - pressure
    - frustration

    Then:
    1. acknowledge the exhaustion directly
    2. reflect the emotional pressure
    3. ask a gentle follow-up question

    GOOD RESPONSE EXAMPLE:
    "That sounds really exhausting. Studying for exams for long periods can drain a lot of energy, especially when you're eager to graduate. What part has been the most overwhelming lately?"

    BAD RESPONSE EXAMPLES:
    - "How are you feeling about school?"
    - "Everything will be okay."
    - "Stay positive."

    USER CONTEXT:
    Emotion: {emotion}
    Risk: {risk}
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

                if response.status_code != 200:
                    logger.error(
                        f"Ollama error {response.status_code}: {response.text}"
                    )
                    continue

                data = response.json()

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

        # --------------------------------------------------
        # CONTEXTUAL EMOTION OVERRIDE (FIXED VIA REGEX)
        # --------------------------------------------------

        msg = message.lower()

        crisis_keywords = [
            "end everything", "hurt myself", "commit suicide", "kill myself", 
            "want to die", "ending seems like", "suicidal", "end this"
        ]

        stress_keywords = [
            "stress", "stressed", "overwhelmed",
            "tired", "exhausted", "burned out",
            "too much studying", "can't study",
            "cant study", "finals", "exams",
            "want to graduate", "graduation pressure",
            "drained"
        ]

        sad_keywords = [
            "lonely", "empty", "sad",
            "depressed", "hopeless",
            "crying", "hurt"
        ]

        anger_keywords = [
            "angry", "mad", "furious",
            "annoyed", "irritated"
        ]

        happy_keywords = [
            "excited", "grateful",
            "happy", "great",
            "amazing"
        ]

        # Priority-based whole word override check
        if contains_whole_word(crisis_keywords, msg):
            emotion = "sad"
            
        elif contains_whole_word(stress_keywords, msg):
            emotion = "anxious"

        elif contains_whole_word(sad_keywords, msg):
            emotion = "sad"

        elif contains_whole_word(anger_keywords, msg):
            emotion = "anger"

        elif contains_whole_word(happy_keywords, msg):
            emotion = "happy"

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

        # Pass through our fixed whole-word safety function
        risk_data = override_risk(message, risk_data, risk_score)

        risk = risk_data["type"]
        risk_score = float(risk_data["score"])

        # -------------------------
        # Mood scoring
        # -------------------------

        mood_map = {
            "happy": 8,
            "calm": 7,
            "neutral": 5,
            "anxious": 3,
            "fear": 3,
            "sad": 2,
            "depressed": 1,
            "anger": 2,
            "angry": 2,
            "stressed": 3
        }

        mood_score = mood_map.get(emotion, 5)

        # -------------------------
        # Session
        # -------------------------

        session = get_or_create_session(
            user_id,
            db,
            message
        )

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

        reply = await generate_reply(
            history,
            message,
            emotion,
            risk
        )

        # -------------------------
        # Save user message
        # -------------------------

        db.execute(
            insert(messages)
            .values(
                session_id=session["id"],
                sender="user",
                message_text=message,
                emotion_label=emotion,
                risk_level=risk,
            )
        )

        # Re-fetch or generate sequence tracking if message ID is mandatory elsewhere
        user_message_id = None 

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
                    message_id=user_message_id,  # Will write null safely if sequence is automatic
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
            "risk": {
                "type": "safe",
                "score": 0
            },
            "mood_score": 5,
        }
