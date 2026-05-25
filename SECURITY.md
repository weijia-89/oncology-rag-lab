# Security

## Reporting a vulnerability

Use GitHub Security Advisories to report vulnerabilities privately:

**https://github.com/weijia-89/oncology-rag-lab/security/advisories/new**

Please include: what you found, steps to reproduce, and what impact you believe it has. You'll get a response within a few days.

---

## Threat model

This is a local developer tool. It has no inbound network surface and no server component, which means the expected runtime is a single developer machine with Ollama running on localhost. That's the whole attack surface.

**What this tool is not:** It's not a web service, API, or multi-user system. It doesn't store credentials, handle authentication, or process real patient data.

### Attack surface in scope

| Surface | Notes |
|---|---|
| Prompt injection via clinical note content | Note text is concatenated into LLM prompts verbatim. A note containing instruction-like text could influence extraction output. See note below. |
| Malicious `.txt` files in `data/` | Files are read with `Path.read_text()`; no execution, no deserialization. Risk is limited to content influencing LLM behavior. |
| ChromaDB files on disk | Written to `data/chroma_db/` (git-ignored). A tampered index could influence retrieval; it is not a remote attack surface. |
| Supply chain (Python dependencies) | `pyproject.toml` pins minimum versions but not exact hashes. `uv lock` would tighten this. |

### Out of scope

- Network-level attacks (no inbound ports opened by this tool)
- Privilege escalation (runs as the invoking user)
- Authentication bypass (no auth exists)

### Prompt injection note

`extract.py` concatenates note text into prompts with structural delimiters (`--- CLINICAL NOTE ---` / `--- END NOTE ---`) but doesn't sanitize note content. Output is validated with `value_within_closed_vocabulary` (forbidden substrings such as `ignore previous`, length bounds) and hijacked answers fall back to `unknown`. Adversarial coverage lives in `tests/eval/test_injection.py` over synthetic `data/injection_notes/` (simulated model compliance cases). Input sanitization and ontology-level schema enforcement remain future production hardening.

---

## Data handling

| Data set | Status |
|---|---|
| `data/synthetic_notes/` | Entirely fictional. No real patient names, dates, providers, or identifiers. |
| `data/edge_case_notes/` | Also fictional. Constructed to exercise specific parsing failure modes. |
| `data/injection_notes/` | Synthetic adversarial injection-pattern notes for eval only. No PHI. |
| `data/gold_standard.csv` | Labels for synthetic notes. No PHI. |
| `data/real_notes/` | **Listed in `.gitignore`.** This directory is never committed. Any files placed there by a developer stay local only. |
| `data/chroma_db/` | Generated index derived from synthetic notes. Git-ignored. |

No HIPAA-covered data has ever been committed to this repository. The `.gitignore` entry for `data/real_notes/` is a deliberate safeguard, not an afterthought.

---

## Dependency notes

Runtime dependencies (see `pyproject.toml`) include `arize-phoenix` and `openinference-instrumentation-llama-index`. Phoenix tracing is opt-in via the `--trace` flag and connects only to `http://localhost:6006` by default, so no data leaves the local machine unless the user overrides `PHOENIX_COLLECTOR_ENDPOINT` to point at a remote collector. That override is on the user; the default is safe.

Ollama communicates only with `OLLAMA_HOST` (default `http://localhost:11434`). No cloud LLM endpoints are used.
