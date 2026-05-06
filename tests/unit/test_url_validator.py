"""Unit tests for the SSRF URL validator in config.py.

We test the validate_service_url function directly. The goal is to confirm:
  - localhost and 127.x addresses always pass (happy path for Ollama)
  - RFC-1918 private addresses pass (LAN Ollama setups)
  - Cloud metadata ranges are blocked (169.254.x.x)
  - Bad schemes raise ValueError
  - Empty input raises ValueError
"""

from __future__ import annotations

import pytest

from onclab.config import validate_service_url

# ---------------------------------------------------------------------------
# Happy-path: addresses that must be allowed
# ---------------------------------------------------------------------------


def test_localhost_http_passes():
    """The default Ollama URL must always be accepted."""
    result = validate_service_url("http://localhost:11434", "OLLAMA_HOST")
    assert result == "http://localhost:11434"


def test_127_loopback_passes():
    result = validate_service_url("http://127.0.0.1:11434", "OLLAMA_HOST")
    assert result == "http://127.0.0.1:11434"


def test_https_localhost_passes():
    result = validate_service_url("https://localhost:6006", "PHOENIX_COLLECTOR_ENDPOINT")
    assert result == "https://localhost:6006"


def test_rfc1918_private_passes():
    """An Ollama instance on a LAN address (192.168.x.x) must be accepted."""
    result = validate_service_url("http://192.168.1.50:11434", "OLLAMA_HOST")
    assert result == "http://192.168.1.50:11434"


def test_rfc1918_10_block_passes():
    result = validate_service_url("http://10.0.0.5:11434", "OLLAMA_HOST")
    assert result == "http://10.0.0.5:11434"


# ---------------------------------------------------------------------------
# Blocked ranges
# ---------------------------------------------------------------------------


def test_cloud_metadata_169_254_blocked():
    """169.254.169.254 is the AWS/GCP/Azure instance metadata endpoint."""
    with pytest.raises(ValueError, match="blocked range"):
        validate_service_url("http://169.254.169.254/latest/meta-data/", "OLLAMA_HOST")


def test_cloud_metadata_169_254_any_blocked():
    """Any address in 169.254.0.0/16 is blocked, not just .169.254."""
    with pytest.raises(ValueError, match="blocked range"):
        validate_service_url("http://169.254.1.1:11434", "OLLAMA_HOST")


# ---------------------------------------------------------------------------
# Bad scheme
# ---------------------------------------------------------------------------


def test_ftp_scheme_rejected():
    with pytest.raises(ValueError, match="scheme"):
        validate_service_url("ftp://localhost:11434", "OLLAMA_HOST")


def test_file_scheme_rejected():
    with pytest.raises(ValueError, match="scheme"):
        validate_service_url("file:///etc/passwd", "OLLAMA_HOST")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string_rejected():
    with pytest.raises(ValueError):
        validate_service_url("", "OLLAMA_HOST")


def test_unresolvable_hostname_rejected():
    with pytest.raises(ValueError, match="could not be resolved"):
        validate_service_url("http://this-hostname-does-not-exist.invalid:11434", "OLLAMA_HOST")
