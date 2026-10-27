"""Unit tests for core helpers: JSON sanitising, password hashing, JWTs."""

import json
import math

import numpy as np
import pytest
from fastapi import HTTPException

from app.core.security import create_access_token, decode_token, hash_password, verify_password
from app.core.utils import json_safe
from app.services.registry_service import default_model_name


class TestJsonSafe:
    def test_nan_and_inf_become_none(self):
        assert json_safe({"a": float("nan"), "b": math.inf, "c": np.float64("nan")}) == {
            "a": None,
            "b": None,
            "c": None,
        }

    def test_numpy_types_converted(self):
        out = json_safe({"i": np.int64(3), "f": np.float32(1.5), "b": np.bool_(True), "arr": np.array([1, 2])})
        assert out == {"i": 3, "f": 1.5, "b": True, "arr": [1, 2]}
        json.dumps(out, allow_nan=False)

    def test_nested_and_non_string_keys(self):
        assert json_safe({1: [{"x": (1, float("nan"))}]}) == {"1": [{"x": [1, None]}]}

    def test_unknown_objects_stringified(self):
        assert json_safe({"o": object})["o"].startswith("<class")


class TestPasswords:
    def test_hash_and_verify(self):
        hashed = hash_password("correct horse")
        assert verify_password("correct horse", hashed)
        assert not verify_password("wrong", hashed)

    def test_malformed_hash_is_rejected_not_raised(self):
        assert not verify_password("x", "not-a-bcrypt-hash")


class TestTokens:
    def test_token_carries_email_and_role(self):
        payload = decode_token(create_access_token("user-1", role="admin", email="a@b.io"))
        assert payload["sub"] == "user-1"
        assert payload["role"] == "admin"
        assert payload["email"] == "a@b.io"

    def test_tampered_token_rejected(self):
        token = create_access_token("user-1")
        with pytest.raises(HTTPException) as exc:
            decode_token(token[:-2] + ("aa" if token[-2:] != "aa" else "bb"))
        assert exc.value.status_code == 401


class TestDefaultModelName:
    def test_slugifies(self):
        assert default_model_name("Credit Card Fraud.csv", "Class") == "credit-card-fraud-csv-class"


class TestDemoUsers:
    def test_demo_accounts_pass_registration_validation(self):
        """Seeded accounts must be loggable — email-validator rejects special-use domains like .local."""
        from app.bootstrap import DEMO_USERS
        from app.schemas.auth import UserCreate

        for email, password, role in DEMO_USERS:
            UserCreate(email=email, password=password, role=role)
