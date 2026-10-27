from pydantic import BaseModel, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    role: str = "engineer"

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v.encode()) > 72:  # bcrypt only uses the first 72 bytes
            raise ValueError("password must be at most 72 bytes")
        return v

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("viewer", "engineer", "admin"):
            raise ValueError("role must be viewer, engineer, or admin")
        return v


class UserLogin(BaseModel):
    email: str  # lookup only — format checks here would just lock existing accounts out
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


class CurrentUser(BaseModel):
    id: str
    email: str | None = None
    role: str
