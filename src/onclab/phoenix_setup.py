"""Arize Phoenix observability bootstrap.

What Phoenix gives you:
    A local OpenTelemetry-compatible tracing UI at http://localhost:6006.
    Every retrieval call, every LLM call, every eval result becomes a span.
    The killer view is "compare two runs side by side" — which is also the
    interview-relevant story: "I run drift detection between pipeline
    versions, traced end-to-end."

Why it lives in its own module (not in main()):
    1. You only want to call register() once per process. Calling it twice
       creates duplicate exporters and you get every span twice in the UI.
       A module-level guard makes this idempotent.
    2. Tests should NOT instrument Phoenix — that would slow them down and
       fill the UI with noise. The guard plus an `enabled=False` toggle
       keeps unit tests clean.
"""

from __future__ import annotations

import os

_INSTRUMENTED = False  # module-level singleton flag


def setup_phoenix(*, project_name: str = "oncology-rag-lab", enabled: bool = True) -> None:
    """Wire Phoenix tracing into llama-index. Idempotent.

    Call once at the top of any CLI command that should be traced. Skip
    in unit tests (pass enabled=False or just don't call).

    Why we don't `phoenix.launch_app()` here: launching the UI server
    is a separate concern. You either:
      - run `make phoenix` in another terminal (the UI is then a long-lived
        process you can leave running across many lab sessions), or
      - call `phoenix.launch_app()` manually in a Jupyter cell.
    The instrumentor connects to whichever Phoenix endpoint is reachable.
    """
    global _INSTRUMENTED
    if _INSTRUMENTED or not enabled:
        return

    # Imports are inside the function so a bare `import onclab` doesn't
    # drag in arize/openinference for users who never call setup_phoenix.
    # This matters at install time more than runtime — keeps the dep tree
    # narrow if Phoenix breaks an install.
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from phoenix.otel import register

    # PHOENIX_COLLECTOR_ENDPOINT is the env var Phoenix's libraries respect.
    # Default points at the local UI started by `make phoenix`.
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")

    tracer_provider = register(project_name=project_name)
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    _INSTRUMENTED = True
