"""Opaque keyset cursors for GET /edges: base64url JSON {detected_at, id}.

The listing orders by (detected_at DESC, id DESC); the cursor carries the
last row's sort key so the next page resumes strictly after it.
"""

import base64
import binascii
import json
import uuid
from dataclasses import dataclass
from datetime import datetime

from agent.api.errors import InvalidParameterError


@dataclass(frozen=True)
class Cursor:
    detected_at: datetime
    id: uuid.UUID


def encode_cursor(detected_at: datetime, edge_id: uuid.UUID) -> str:
    payload = json.dumps({"detected_at": detected_at.isoformat(), "id": str(edge_id)})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> Cursor:
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()))
        return Cursor(detected_at=datetime.fromisoformat(payload["detected_at"]), id=uuid.UUID(payload["id"]))
    except (binascii.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise InvalidParameterError("Invalid pagination cursor") from exc
