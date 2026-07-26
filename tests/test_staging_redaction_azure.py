"""Tests for the additional Azure secret redaction patterns in staging.

``redact_secrets`` scrubs credential-looking substrings before persisting any
free text to the staging directory. This covers the newly added Azure SAS
signature, storage account key, and connection-string password patterns.
"""
from __future__ import annotations

from skillopt_sleep.staging import redact_secrets


def test_azure_sas_signature_redacted() -> None:
    url = "https://acct.blob.core.windows.net/c/b?sig=abcDEF123%2Bxyz789QQ&se=2026"
    out = redact_secrets(url)
    assert "[REDACTED_SAS_SIG]" in out
    assert "abcDEF123" not in out


def test_storage_account_key_redacted() -> None:
    conn = "DefaultEndpointsProtocol=https;AccountKey=aB3dEfGhIjKlMnOpQrStUvWx==;"
    out = redact_secrets(conn)
    assert "[REDACTED_STORAGE_KEY]" in out
    assert "aB3dEfGhIjKlMnOpQrStUvWx" not in out


def test_connection_string_password_redacted() -> None:
    conn = "Server=db;Password=Sup3rSecret!;Database=app"
    out = redact_secrets(conn)
    assert "[REDACTED_DB_PASS]" in out
    assert "Sup3rSecret" not in out
    assert out == "Server=db;Password=[REDACTED_DB_PASS];Database=app"


def test_quoted_connection_string_password_redacted() -> None:
    conn = 'Server=db;Password="Sup3r; Secret!";Database=app'
    out = redact_secrets(conn)
    assert out == "Server=db;Password=[REDACTED_DB_PASS];Database=app"


def test_generic_secret_redaction_preserves_following_fields() -> None:
    text = "token=top-secret&request=42;status=failed"
    out = redact_secrets(text)
    assert out == "token=[REDACTED]&request=42;status=failed"


def test_recurses_into_containers() -> None:
    payload = {"logs": ["ok", "AccountKey=aB3dEfGhIjKlMnOpQrStUvWx=="]}
    out = redact_secrets(payload)
    assert "[REDACTED_STORAGE_KEY]" in out["logs"][1]


def test_plain_text_unchanged() -> None:
    assert redact_secrets("the quick brown fox jumps") == "the quick brown fox jumps"
