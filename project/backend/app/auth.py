from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .schemas import User


security = HTTPBearer(auto_error=False)


def _build_user(username: str) -> User:
    return User(id=1, username=username, displayName=username, roles=["admin"])


def authenticate_user(username: str, password: str) -> Optional[User]:
    if username == settings.admin_username and password == settings.admin_password:
        return _build_user(username)
    return None


def _create_token(user: User, token_type: str, expires_in: int) -> str:
    payload: Dict[str, Any] = {
        "sub": str(user.id),
        "username": user.username,
        "roles": user.roles,
        "type": token_type,
        "exp": datetime.utcnow() + timedelta(seconds=expires_in),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def create_access_token(user: User) -> str:
    return _create_token(user, "access", settings.jwt_expire_seconds)


def create_refresh_token(user: User) -> str:
    return _create_token(user, "refresh", settings.jwt_refresh_seconds)


def decode_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    try:
        payload = decode_token(credentials.credentials)
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return _build_user(payload.get("username", "user"))
