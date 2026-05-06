# Security

## Reporting a vulnerability

Use GitHub Security Advisories to report vulnerabilities privately:

**https://github.com/weijia-89/oncology-rag-lab/security/advisories/new**

Please include: what you found, steps to reproduce, and what impact you believe it has. You will get a response within a few days.

---

## Threat model

This is a local developer tool. It has no inbound network surface and no server component. The expected runtime is a single developer machine with Ollama running on localhost.

**What this tool is not:** It is not a web service, API, or multi-user system. It does not store credentials, handle authentication, or process real patient data.

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

`extract.py` concatenates note text into prompts with structural delimiters (`--- CLINICAL NOTE ---` / `--- END NOTE ---`) but does not sanitize the content. All committed notes in `data/synthetic_notes/` and `data/edge_case_notes/` are invented and safe. The file-path comment in `extract.py` describes what a production hardening pass would add (input sanitization, closed-vocabulary output validation, adversarial test cases in the eval suite).

---

## Data handling

| Data set | Status |
|---|---|
| `data/synthetic_notes/` | Entirely fictional. No real patient names, dates, providers, or identifiers. |
| `data/edge_case_notes/` | Also fictional. Constructed to exercise specific parsing failure modes. |
| `data/gold_standard.csv` | Labels for synthetic notes. No PHI. |
| `data/real_notes/` | **Listed in `.gitignore`.** This directory is never committed. Any files placed there by a developer stay local only. |
| `data/chroma_db/` | Generated index derived from synthetic notes. Git-ignored. |

No HIPAA-covered data has ever been committed to this repository. The `.gitignore` entry for `data/real_notes/` is a deliberate safeguard, not an afterthought.

---

## Dependency notes

Runtime dependencies (see `pyproject.toml`) include `arize-phoenix` and `openinference-instrumentation-llama-index`. Phoenix tracing is opt-in (`--trace` flag) and connects only to `http://localhost:6006` by default. No data leaves the local machine unless the user overrides `PHOENIX_COLLECTOR_ENDPOINT` to point at a remote collector.

Ollama communicates only with `OLLAMA_HOST` (default `http://localhost:11434`). No cloud LLM endpoints are used.
