"""Cursor codec: base64url JSON keyset {detected_at, id}."""

import base64
import uuid
from datetime import UTC, datetime

import pytest

from agent.api.errors import InvalidParameterError
from agent.api.pagination import decode_cursor, encode_cursor


class TestCursorCodec:
    def test_roundtrip(self) -> None:
        detected_at = datetime(2026, 7, 4, 12, 30, 45, tzinfo=UTC)
        edge_id = uuid.uuid4()
        cursor = decode_cursor(encode_cursor(detected_at, edge_id))
        assert cursor.detected_at == detected_at
        assert cursor.id == edge_id

    def test_cursor_is_opaque_base64url(self) -> None:
        encoded = encode_cursor(datetime(2026, 7, 4, tzinfo=UTC), uuid.uuid4())
        decoded = base64.urlsafe_b64decode(encoded.encode()).decode()
        assert '"detected_at"' in decoded
        assert '"id"' in decoded

    def test_garbage_rejected(self) -> None:
        with pytest.raises(InvalidParameterError):
            decode_cursor("not-a-cursor!!!")

    def test_valid_base64_wrong_payload_rejected(self) -> None:
        bogus = base64.urlsafe_b64encode(b'{"nope": 1}').decode()
        with pytest.raises(InvalidParameterError):
            decode_cursor(bogus)

    def test_non_json_payload_rejected(self) -> None:
        bogus = base64.urlsafe_b64encode(b"plain text").decode()
        with pytest.raises(InvalidParameterError):
            decode_cursor(bogus)
