from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.deps import get_current_user
from app.models.app_user import AppUser
from app.rate_limit import limiter
from app.schemas.common import LoginRequest, TokenResponse
from app.security.auth import create_access_token, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
@limiter.limit("10/minute")
async def login(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        body = LoginRequest.model_validate(await request.json())
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid login payload") from None
    user = (await db.execute(select(AppUser).where(AppUser.email == body.email.lower().strip()))).scalar_one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.id, user.role, extra={"email": user.email})
    return TokenResponse(
        access_token=token, role=user.role, email=user.email, display_name=user.display_name
    )


@router.get("/me")
async def me(user: AppUser = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "role": user.role, "display_name": user.display_name}
