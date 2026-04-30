from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MoodCreate(BaseModel):
    user_id: int
    mood_score: int
    notes: Optional[str] = None


class MoodResponse(BaseModel):
    id: int
    user_id: int
    mood_score: int
    notes: Optional[str]
    emotion_label: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True