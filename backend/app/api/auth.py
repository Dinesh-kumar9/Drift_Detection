"""Auth API — Phase 1: register + login with JWT."""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import create_access_token, hash_password, verify_password
from app.models import User
from app.schemas.auth import TokenResponse, UserCreate, UserLogin

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create a new user account and return a JWT token."""
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalars().first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        id=str(uuid.uuid4()),
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        created_at=datetime.now(timezone.utc),
    )
    db.add(user)
    await db.flush()

    token = create_access_token(subject=user.id, role=user.role)
    return TokenResponse(access_token=token, role=user.role, email=user.email)


@router.post("/login", response_model=TokenResponse)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate and return a JWT token."""
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalars().first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(subject=user.id, role=user.role)
    return TokenResponse(access_token=token, role=user.role, email=user.email)


@router.get("/me")
async def me(
    current_user: dict = Depends(__import__("app.core.security", fromlist=["get_current_user"]).get_current_user)
):
    return current_user
