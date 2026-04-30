from datetime import datetime

from sqlalchemy import (Column, DateTime, ForeignKey, Integer, MetaData,
                        String, Table, Text)

metadata = MetaData()

# ---------------------------
# USERS
# ---------------------------
users = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("email", String, unique=True, nullable=False),
    Column("password_hash", String, nullable=False),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# ---------------------------
# CHAT SESSIONS
# ---------------------------
chat_sessions = Table(
    "chat_sessions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("title", String, default="New Chat"),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# ---------------------------
# MESSAGES
# ---------------------------
messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("session_id", Integer, ForeignKey("chat_sessions.id")),
    Column("sender", String),   # user / bot
    Column("message_text", Text),
    Column("emotion_label", String),
    Column("risk_level", String),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# ---------------------------
# MOOD TRACKING
# ---------------------------
mood_tracking = Table(
    "mood_tracking",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("mood_score", Integer),
    Column("emotion_label", String),
    Column("notes", Text),
    Column("created_at", DateTime, default=datetime.utcnow)
)

# ---------------------------
# ALERTS
# ---------------------------
alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("message_id", Integer, ForeignKey("messages.id")),
    Column("risk_level", String),
    Column("trigger_text", Text),
    Column("created_at", DateTime, default=datetime.utcnow)
)