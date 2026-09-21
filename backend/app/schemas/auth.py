from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    role: str = "engineer"

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("viewer", "engineer", "admin"):
            raise ValueError("role must be viewer, engineer, or admin")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class UserResponse(BaseModel):
    id: str
    email: str
    role: str

    model_config = {"from_attributes": True}
