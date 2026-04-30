from pydantic import BaseModel


class RiskRequest(BaseModel):
    text: str

class RiskResponse(BaseModel):
    status: str
    score: float
