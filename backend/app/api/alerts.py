from fastapi import APIRouter, Depends
from sqlalchemy import select

from backend.app.db.database import get_db
from backend.app.db.models import alerts
from backend.app.utils.dependencies import get_current_user

router = APIRouter(
    prefix="/alerts",
    tags=["Alerts"]
)


@router.get("/")
def get_alerts(
    user=Depends(get_current_user),
    db=Depends(get_db)
):
    result = db.execute(
        select(alerts)
        .where(alerts.c.user_id == int(user["user_id"]))
        .order_by(alerts.c.id.desc())
    ).fetchall()

    return [
        {
            "id": row._mapping["id"],
            "risk_level": row._mapping["risk_level"],
            "trigger_text": row._mapping["trigger_text"],
            "created_at": row._mapping["created_at"]
        }
        for row in result
    ]