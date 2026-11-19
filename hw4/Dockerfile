# Replace this named context with a local wheelhouse for an offline build.
FROM scratch AS wheels
COPY hw4/requirements.txt /

FROM docker.io/library/python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
WORKDIR /app

COPY hw4/requirements.txt /tmp/freshqa-mcp-requirements.txt
ARG FRESHQA_MCP_OFFLINE=0
RUN --mount=type=bind,from=wheels,target=/opt/wheels \
    if [ "$FRESHQA_MCP_OFFLINE" = 1 ]; then set -- --no-index; else set --; fi; \
    python -m pip --isolated --disable-pip-version-check install --no-cache-dir \
        --find-links=/opt/wheels "$@" -r /tmp/freshqa-mcp-requirements.txt

COPY freshqa/__init__.py freshqa/dataset.py freshqa/official_sources.py freshqa/quantities.py freshqa/source_diagnostics.py freshqa/tools.py freshqa/
COPY hw2/lexical_baseline.py hw2/prepare_corpus.py hw2/
COPY hw3/__init__.py hw3/snapshot_tools.py hw3/
COPY hw4/mcp_server.py hw4/
COPY hw4/data/corpus.jsonl hw4/data/

USER 65534:65534
ENTRYPOINT ["python", "-m", "hw4.mcp_server"]
