"""Debug guest authentication dependency."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User

DEBUG_GUEST_HEADER = "X-Debug-Guest-Id"
DEVICE_FINGERPRINT_HEADER = "X-Debug-Device-Fingerprint"


def _clean_header(value: str | None, max_len: int) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return value[:max_len]


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    debug_guest_id: str | None = Header(default=None, alias=DEBUG_GUEST_HEADER),
    device_fingerprint: str | None = Header(
        default=None, alias=DEVICE_FINGERPRINT_HEADER
    ),
) -> User:
    """Resolve or create the current debug guest user from request headers."""
    guest_id = _clean_header(debug_guest_id, 128)
    if guest_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing {DEBUG_GUEST_HEADER}",
        )

    fingerprint = _clean_header(device_fingerprint, 128)
    user_agent = _clean_header(request.headers.get("user-agent"), 2000)

    user = db.query(User).filter(User.debug_guest_id == guest_id).one_or_none()
    if user is None:
        user = User(
            debug_guest_id=guest_id,
            device_fingerprint=fingerprint,
            user_agent=user_agent,
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            user = db.query(User).filter(User.debug_guest_id == guest_id).one()
        else:
            db.refresh(user)
    else:
        user.device_fingerprint = fingerprint or user.device_fingerprint
        user.user_agent = user_agent or user.user_agent
        user.last_seen_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(user)

    return user
