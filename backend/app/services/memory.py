# services/memory.py
from sqlalchemy import select
from backend.app.db.models import messages


def get_short_term_memory(session_id, db, limit=8):
    """
    Returns last N messages from a session as a list of dicts
    compatible with normalize_history and the Ollama message format.
    """

    rows = db.execute(
        select(messages)
        .where(messages.c.session_id == session_id)
        .order_by(messages.c.id.desc())
        .limit(limit)
    ).fetchall()

    # reverse to chronological order
    rows = rows[::-1]

    history = []
    for r in rows:
        role = "assistant" if r.sender == "bot" else "user"
        history.append({
            "role": role,
            "content": r.message_text
        })

    return history
