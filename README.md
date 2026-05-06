# onclab — local oncology RAG testbed

[![CI](https://github.com/weijia-89/oncology-rag-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/weijia-89/oncology-rag-lab/actions/workflows/ci.yml)

A working RAG pipeline that pulls structured oncology entities (cancer type, AJCC stage, regimen, ECOG, ...) out of synthetic clinical notes. The point is not the entities; the point is the testing infrastructure around them. Same patterns production oncology pipelines use at 150M-document scale, on a single machine.

## What's in the box

| Layer | Tool | Where it lives |
|---|---|---|
| Inference | Ollama (qwen3:14b) | local server at :11434 |
| Embedding | nomic-embed-text via Ollama | `src/onclab/ingest.py` |
| Vector store | ChromaDB (file-based) | `data/chroma_db/` |
| RAG framework | LlamaIndex | `src/onclab/rag.py` |
| Entity extraction | pydantic schema + per-entity prompts | `src/onclab/extract.py` |
| Eval | DeepEval (pytest-native) | `tests/eval/` |
| Observability | Arize Phoenix | `src/onclab/phoenix_setup.py`, UI at :6006 |
| Drift detection | A/B model comparison | `scripts/drift_compare.py` |

## Quickstart (fresh box)

```bash
# 1. Install Ollama (https://ollama.com), then start it.
# 2. From this directory:
scripts/bootstrap.sh         # uv sync + ollama pull qwen3:14b + nomic-embed-text
make ingest                  # chunk + embed + index 8 synthetic notes
make extract                 # see the structured outputs for each note
```

To run the eval suite end-to-end:

```bash
make eval                    # DeepEval-driven tests with real Ollama calls (slow)
make check-regression        # gate: did pass rate drop >5% vs baseline?
```

For unit tests with no Ollama dependency:

```bash
make test-unit               # MOCK_LLM=1; runs in seconds; CI-friendly
```

## Read this in order

For learning the layers, read the files in the order the data flows through them:

1. `src/onclab/config.py` — every knob in one place
2. `src/onclab/llm_client.py` — Ollama wrapper + MOCK_LLM toggle
3. `src/onclab/ingest.py` — load → chunk → embed → store
4. `src/onclab/rag.py` — retrieval + query engine
5. `src/onclab/extract.py` — the actual oncology task
6. `tests/eval/test_retrieval.py` — deterministic-layer tests
7. `tests/eval/test_extraction_eval.py` — DeepEval scoring
8. `scripts/drift_compare.py` — side-by-side model diff

Every file has comments at the top explaining the design decision and what the alternative looked like.

## Hardware notes

This was sized for an RTX 4070 Ti Super (16 GB VRAM) with 32 GB system RAM. Qwen3 14B Q4_K_M fits at ~8.7 GB and runs at ~30–40 tokens/sec. If you only have 8 GB VRAM, swap to llama3.1:8b in `.env`.

vLLM is intentionally not used. Ollama is faster to set up, restarts cleanly on driver updates, and has no meaningful throughput advantage for single-user batch eval at this scale.

## Design questions worth thinking through

- What changes if the corpus is 150M documents instead of 8? (Chunking strategy, embedding batch size, ChromaDB → managed vector store, async ingest pipeline.)
- How does test design change when FDA sensitivity/specificity replaces internal SLAs? (Weighted precision/recall metrics in GEval; stricter false-negative budget for critical entities like `ajcc_stage`.)
- Where does the audit trail live in this design? (Every retrieval call traces to Phoenix; every entity carries a rationale string.)

## Files you might add later

- `data/synthetic_notes/patient_009.txt` and onward — `scripts/seed_data.py` shows the template pattern.
- A second embedding model (BGE-M3) for retrieval quality A/B — `OllamaEmbedding` -> `HuggingFaceEmbedding`, one swap in `ingest.py`.
- A custom GEval metric for clinical-plausibility scoring — DeepEval supports it; goes alongside `test_extraction_eval.py`.

## License / Provenance

All clinical content in `data/synthetic_notes/` is invented. No PHI. No real patient ever passed through this code.
