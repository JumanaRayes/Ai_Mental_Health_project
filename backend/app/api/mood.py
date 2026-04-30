from datetime import datetime, timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from backend.app.db.database import get_db
from backend.app.db.models import mood_tracking
from backend.app.utils.dependencies import get_current_user

router = APIRouter(prefix="/mood", tags=["Mood Tracking"])


# DAILY LAST 7 DAYS
@router.get("/daily")
def get_daily_mood(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    last_7 = datetime.utcnow() - timedelta(days=7)

    result = db.execute(
        select(
            func.date(mood_tracking.c.created_at),
            func.avg(mood_tracking.c.mood_score)
        )
        .where(mood_tracking.c.user_id == int(user["user_id"]))
        .where(mood_tracking.c.created_at >= last_7)
        .group_by(func.date(mood_tracking.c.created_at))
        .order_by(func.date(mood_tracking.c.created_at))
    ).fetchall()

    return [
        {
            "date": str(row[0]),
            "avg_mood_score": round(float(row[1]), 2)
        }
        for row in result
    ]


# WEEKLY AVG LAST 4 WEEKS
@router.get("/weekly")
def get_weekly_mood(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    result = db.execute(
        select(
            func.date_trunc("week", mood_tracking.c.created_at),
            func.avg(mood_tracking.c.mood_score)
        )
        .where(mood_tracking.c.user_id == int(user["user_id"]))
        .group_by(func.date_trunc("week", mood_tracking.c.created_at))
        .order_by(func.date_trunc("week", mood_tracking.c.created_at))
    ).fetchall()

    return [
        {
            "week": str(row[0].date()),
            "avg_mood_score": round(float(row[1]), 2)
        }
        for row in result
    ]


# EMOTION DISTRIBUTION
@router.get("/stats")
def emotion_stats(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    result = db.execute(
        select(
            mood_tracking.c.emotion_label,
            func.count()
        )
        .where(mood_tracking.c.user_id == int(user["user_id"]))
        .group_by(mood_tracking.c.emotion_label)
    ).fetchall()

    return [
        {
            "emotion": row[0],
            "count": row[1]
        }
        for row in result
    ]