from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class User:
    id: int
    username: str
    is_admin: bool = False


class UserJsonError(ValueError):
    pass


def parse_user_json(payload: str) -> User:
    """Safely parse user JSON with strict validation.

    Fixes security bugs:
    - Uses `json.loads` not `eval` to avoid code execution
    - Validates required fields and types
    - Rejects unexpected extra fields to prevent mass-assignment
    - Coerces booleans but rejects truthy strings like "true"
    """
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise UserJsonError(f"Invalid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise UserJsonError("Top-level JSON must be an object")

    allowed_keys = {"id", "username", "is_admin"}
    missing = {"id", "username"} - obj.keys()
    if missing:
        raise UserJsonError(f"Missing required fields: {sorted(missing)}")

    extra = set(obj.keys()) - allowed_keys
    if extra:
        raise UserJsonError(f"Unexpected fields are not allowed: {sorted(extra)}")

    user_id = obj["id"]
    if not isinstance(user_id, int) or user_id < 0:
        raise UserJsonError("id must be a non-negative integer")

    username = obj["username"]
    if not isinstance(username, str) or not username:
        raise UserJsonError("username must be a non-empty string")

    is_admin = obj.get("is_admin", False)
    if not isinstance(is_admin, bool):
        raise UserJsonError("is_admin must be a boolean if provided")

    return User(id=user_id, username=username, is_admin=is_admin)
