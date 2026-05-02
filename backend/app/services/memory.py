from sqlalchemy import select
from backend.app.db.models import messages


def get_short_term_memory(session_id, db, limit=8):
    """
    Returns last N messages from a session as readable conversation history.
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
        role = "User" if r.sender == "user" else "Assistant"
        history.append(f"{role}: {r.message_text}")

    return "\n".join(history)