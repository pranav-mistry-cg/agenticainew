from datetime import datetime
from enum import Enum
from typing import Annotated, List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Path, Body, status
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, HttpUrl, EmailStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

app = FastAPI(title="Pydantic + FastAPI Demo", version="1.0.0")

# -----------------------------
# 1) Settings via Pydantic (v2)
# -----------------------------
class Settings(BaseSettings):
    # Load from ENV vars or .env automatically if present
    APP_NAME: str = "pydantic-fastapi"
    DEFAULT_PAGE_SIZE: int = 20
    MAX_PAGE_SIZE: int = 100

     # Pydantic v2 style config (replaces inner class Config)
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> Settings:
    # You could cache this if needed
    return Settings()


# -----------------------------
# 2) Domain Models (Pydantic)
# -----------------------------
class Role(str, Enum):
    admin = "admin"
    viewer = "viewer"
    editor = "editor"


class UserBase(BaseModel):
    # Pydantic v2 Field with constraints:
    name: str = Field(..., min_length=2, max_length=80, examples=["Pranav"])
    email: EmailStr
    website: Optional[HttpUrl] = None
    role: Role = Role.viewer


class UserCreate(UserBase):
    password: str = Field(..., min_length=8, examples=["S3curePass!"])


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=80)
    website: Optional[HttpUrl] = None
    role: Optional[Role] = None


class User(UserBase):
    id: int
    created_at: datetime


class UsersPage(BaseModel):
    items: List[User]
    total: int
    page: int
    size: int


# -----------------------------
# 3) Fake in-memory store
# -----------------------------
DB: dict[int, User] = {}
SEQ = 0


def create_user(payload: UserCreate) -> User:
    global SEQ
    SEQ += 1
    user = User(
        id=SEQ,
        name=payload.name,
        email=payload.email,
        website=payload.website,
        role=payload.role,
        created_at=datetime.utcnow(),
    )
    DB[user.id] = user
    return user


def update_user(uid: int, changes: UserUpdate) -> User:
    if uid not in DB:
        raise KeyError(uid)
    existing = DB[uid]

    # Use model_copy(update=...) to merge changes (Pydantic v2)
    merged = existing.model_copy(
        update={
            k: v for k, v in changes.model_dump(exclude_unset=True).items()
        }
    )
    DB[uid] = merged
    return merged


# -----------------------------
# 4) Dependency-injected config
# -----------------------------
def pagination_defaults(
    settings: Annotated[Settings, Depends(get_settings)],
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[Optional[int], Query(ge=1)] = None,
):
    sz = size or settings.DEFAULT_PAGE_SIZE
    return page, min(sz, settings.MAX_PAGE_SIZE)


# -----------------------------
# 5) Routes using Pydantic models
# -----------------------------
@app.post("/users", response_model=User, status_code=status.HTTP_201_CREATED)
def create_user_endpoint(
    payload: Annotated[UserCreate, Body(..., description="User payload")]
):
    try:
        return create_user(payload)
    except ValidationError as e:
        # Typically FastAPI handles model validation automatically;
        # this is just an example if you validate manually.
        raise HTTPException(status_code=422, detail=jsonable_encoder(e.errors()))


@app.get("/users/{user_id}", response_model=User)
def get_user(user_id: Annotated[int, Path(ge=1)]):
    user = DB.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.get("/users", response_model=UsersPage)
def list_users(
    page_size: Annotated[tuple[int, int], Depends(pagination_defaults)],
):
    page, size = page_size
    items = list(DB.values())
    total = len(items)

    # simple pagination
    start = (page - 1) * size
    end = start + size
    slice_ = items[start:end]

    return UsersPage(items=slice_, total=total, page=page, size=size)


@app.patch("/users/{user_id}", response_model=User)
def patch_user(
    user_id: Annotated[int, Path(ge=1)],
    changes: UserUpdate = Body(...),
):
    try:
        return update_user(user_id, changes)
    except KeyError:
        raise HTTPException(status_code=404, detail="User not found")


@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: Annotated[int, Path(ge=1)]):
    if user_id not in DB:
        raise HTTPException(status_code=404, detail="User not found")
    del DB[user_id]
    return None