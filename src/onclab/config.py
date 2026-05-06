"""Centralized config.

Why a config module instead of magic strings inside each file:
    Every model name, path, or threshold gets read from environment variables
    with a sensible default. Change a default here, and the entire pipeline
    picks it up. This is the same pattern Ontada's pipeline almost certainly
    uses for model versioning — no hard-coded "qwen3:14b" sprinkled across
    20 files when you want to A/B test against a new model.

Why pydantic-settings (or in this case, a plain dataclass with os.environ):
    Pydantic validates types at load time. We're using a plain dataclass +
    os.getenv to keep the dep surface small; if config grows complicated,
    swap to pydantic-settings without changing the public API.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# URL / SSRF validation
# ---------------------------------------------------------------------------
# Cloud-metadata addresses that must never be contacted from an LLM pipeline.
# These are IANA-reserved or well-known metadata ranges; connecting to them
# from an application is almost always a sign of an SSRF attack.
_BLOCKED_NETWORKS: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / AWS/GCP/Azure IMDS
    ipaddress.ip_network("100.64.0.0/10"),   # shared address space (carrier-grade NAT)
]


def validate_service_url(url: str, var_name: str) -> str:
    """Validate a service URL before it touches the network.

    Rules:
    - Scheme must be http or https.
    - Hostname resolves and is not in a cloud-metadata range.
    - Loopback (127.x, ::1) and RFC-1918 addresses are always allowed —
      Ollama running on localhost is the expected happy path.

    Raises ValueError if the URL is invalid or blocked.
    Returns the url unchanged if it passes.
    """
    if not url:
        raise ValueError(f"{var_name} must not be empty.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"{var_name}={url!r} uses scheme {parsed.scheme!r}; only http/https are allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"{var_name}={url!r} has no resolvable hostname.")

    # Resolve to an IP. socket.getaddrinfo handles both IPv4 and IPv6 and
    # also covers hostnames like "localhost" that are not raw IPs.
    try:
        results = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise ValueError(
            f"{var_name}={url!r} hostname {hostname!r} could not be resolved: {exc}"
        ) from exc

    for _family, _type, _proto, _canonname, sockaddr in results:
        addr_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(addr_str)
        except ValueError:
            continue

        # Block cloud-metadata ranges first — these take precedence over the
        # is_private check because on Python <=3.10 link-local (169.254.0.0/16)
        # is included in is_private, and we must never allow IMDS addresses
        # regardless of how the stdlib classifies them.
        for blocked in _BLOCKED_NETWORKS:
            if ip in blocked:
                raise ValueError(
                    f"{var_name}={url!r} resolves to {ip}, which is in the "
                    f"blocked range {blocked}. This range is reserved for cloud "
                    "instance metadata and must not be contacted by the pipeline."
                )

        # Loopback (127.x, ::1) and RFC-1918 private addresses are allowed —
        # this covers localhost Ollama and LAN-hosted Ollama instances.
        if ip.is_loopback or ip.is_private:
            continue

        # Anything that reaches here is a public routable IP. The pipeline
        # only talks to local or LAN services; a public IP in OLLAMA_HOST or
        # CHROMA_HOST is either a misconfiguration or an SSRF attempt.
        raise ValueError(
            f"{var_name}={url!r} resolves to a public routable IP ({ip}). "
            "Only localhost and RFC-1918 addresses are permitted for pipeline "
            "services. Check OLLAMA_HOST / CHROMA_HOST in your .env."
        )

    return url


@dataclass(frozen=True)
class Settings:
    """Read once at import time; immutable for the rest of the process.

    `frozen=True` means you can't accidentally mutate `settings.llm_model`
    halfway through a pipeline run. If you need to swap models for drift
    testing, you build a *new* Settings instance and pass it explicitly —
    there's no global mutation, which makes the drift comparison test honest.
    """

    # ---- Connection ----
    ollama_host: str
    """URL of the local Ollama server. Default: http://localhost:11434."""

    # ---- Models ----
    llm_model: str
    """Name of the LLM model in Ollama (e.g., 'qwen3:14b')."""

    embed_model: str
    """Name of the embedding model in Ollama (e.g., 'nomic-embed-text')."""

    # ---- Storage ----
    persist_dir: Path
    """Where ChromaDB writes its files. Resets on `make clean`."""

    notes_dir: Path
    """Where the synthetic clinical notes live."""

    # ---- Retrieval knobs ----
    chunk_size: int = 500
    """Tokens (well, characters via the recursive splitter) per chunk.
    500 is a reasonable default for clinical-style notes — long enough
    to keep a paragraph intact, short enough that retrieval doesn't return
    huge slabs of irrelevant text. Smaller chunks = more precise retrieval
    but more chunks to manage; larger = the opposite."""

    chunk_overlap: int = 50
    """Characters of overlap between adjacent chunks. Overlap exists so
    that a fact straddling a chunk boundary doesn't get cut in half. 10%
    of chunk_size is a common rule of thumb."""

    top_k: int = 3
    """Number of chunks to retrieve per query. With only 8 small notes
    you don't need a large top_k. In production-scale corpora you'd tune
    this empirically against retrieval-precision metrics."""

    # ---- Eval thresholds ----
    # These are the "what counts as passing" bars for the DeepEval suite.
    # Setting them in config (not in test code) is so that you can tighten
    # them as the pipeline matures without editing test files.
    hallucination_threshold: float = 0.5
    faithfulness_threshold: float = 0.7
    answer_relevancy_threshold: float = 0.7

    # ---- Behavior toggles ----
    mock_llm: bool = False
    """When True, llm_client returns canned responses instead of calling Ollama.
    Lets unit tests run in CI without a GPU box."""

    phoenix_project: str = "oncology-rag-lab"


def load_settings(*, persist_dir: Path | None = None, notes_dir: Path | None = None) -> Settings:
    """Construct Settings from environment variables.

    Pattern: explicit overrides > env vars > defaults.
    The kwargs let CLI commands force a specific path without touching env.
    """
    repo_root = Path(__file__).resolve().parents[2]  # src/onclab/config.py -> ../.. = project root

    ollama_host = validate_service_url(
        os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        "OLLAMA_HOST",
    )
    phoenix_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    if phoenix_endpoint:
        validate_service_url(phoenix_endpoint, "PHOENIX_COLLECTOR_ENDPOINT")

    return Settings(
        ollama_host=ollama_host,
        llm_model=os.getenv("ONCLAB_LLM_MODEL", "qwen3:14b"),
        embed_model=os.getenv("ONCLAB_EMBED_MODEL", "nomic-embed-text"),
        persist_dir=persist_dir or (repo_root / "data" / "chroma_db"),
        notes_dir=notes_dir or (repo_root / "data" / "synthetic_notes"),
        mock_llm=os.getenv("MOCK_LLM", "0") == "1",
        phoenix_project=os.getenv("PHOENIX_PROJECT_NAME", "oncology-rag-lab"),
    )
