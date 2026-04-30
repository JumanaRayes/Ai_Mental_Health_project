from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import insert, select

from backend.app.db.database import get_db
from backend.app.db.models import users
from backend.app.schemas.auth_schema import UserRegister
from backend.app.utils.security import (create_access_token, hash_password,
                                        verify_password)

router = APIRouter(tags=["Auth"])


# -----------------------------------
# REGISTER
# -----------------------------------
@router.post("/register")
def register(user: UserRegister, db=Depends(get_db)):

    existing = db.execute(
        select(users).where(users.c.email == user.email)
    ).first()

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already exists"
        )

    db.execute(
        insert(users).values(
            email=user.email,
            password_hash=hash_password(user.password)
        )
    )

    db.commit()

    return {"message": "User created successfully"}


# -----------------------------------
# LOGIN
# -----------------------------------
@router.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db)
):

    result = db.execute(
        select(users).where(
            users.c.email == form_data.username
        )
    ).first()

    if not result:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    db_user = result._mapping

    if not verify_password(
        form_data.password,
        db_user["password_hash"]
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token({
        "user_id": str(db_user["id"]),
        "email": db_user["email"],
        "role": "user"
    })

    return {
        "access_token": token,
        "token_type": "bearer"
    }