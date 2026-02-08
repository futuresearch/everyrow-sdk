# Testing

## Quick Reference

```bash
# SDK unit tests (always run before committing)
uv run pytest tests/ -v

# MCP server tests
cd everyrow-mcp && uv run pytest tests/ -v

# Shell hook tests (require jq)
bash everyrow-mcp/tests/test_statusline.sh
bash everyrow-mcp/tests/test_hook_stop_guard.sh
bash everyrow-mcp/tests/test_hook_results.sh

# Integration tests (require EVERYROW_API_KEY, costs $5-15)
uv run --env-file .env pytest -m integration tests/integration/ -v
```

## What Runs in CI

The `checks.yml` workflow runs on every PR and push to main:

| Job | What it runs | Time |
|-----|-------------|------|
| `lint` | `ruff check`, `ruff format --check`, `basedpyright` | ~30s |
| `test` | `pytest tests/` — SDK unit tests (test_ops, test_task, test_version) | ~5s |
| `mcp-test` | `pytest everyrow-mcp/tests/` — MCP server tests (test_server, test_utils) | ~5s |
| `shell-tests` | Shell hook tests (statusline, stop guard, results) | ~2s |

## Integration Tests

Integration tests make real API calls and cost money (~$5-15 per full run). They are **not** run automatically on PRs.

**How to run:**
```bash
export EVERYROW_API_KEY=your_key_here
uv run pytest -m integration tests/integration/ -v
```

**When to run:**
- Before releases
- After significant changes to SDK operations or the API client
- Via `workflow_dispatch` on the `integration-tests.yml` workflow (requires `EVERYROW_API_KEY` secret in GitHub)

**What they test:**
- `test_agent_map.py` — agent_map: return types, custom response models, column preservation
- `test_screen.py` — screen: passes field, filtering, custom response models
- `test_rank.py` — rank: ascending/descending sort, custom response models
- `test_dedupe.py` — dedupe: equivalence, duplicate identification
- `test_merge.py` — merge: joined tables, subsidiary matching, fuzzy names
- `test_single_agent.py` — single_agent: scalar result, table output, DataFrame input

**MCP integration tests:**
```bash
cd everyrow-mcp
RUN_INTEGRATION_TESTS=1 EVERYROW_API_KEY=your_key uv run pytest tests/test_integration.py -v
```

## Verification Scripts

These are **not** automated tests. They're manual verification tools for specific scenarios:

| Script | When to use |
|--------|------------|
| `scripts/test_progress_e2e.py` | Verify progress output works end-to-end against a local engine |
| `scripts/test_progress_verbose.py` | Debug why intermediate progress counts aren't changing |
| `everyrow-mcp/scripts/verify_sdk_progress.py` | Verify `~/.everyrow/progress.jsonl` was emitted incrementally |
| `everyrow-mcp/scripts/verify_transcript.sh` | Parse a Claude Code session JSONL to verify submit→progress→results tool sequence |

**Running against a local engine:**
```bash
# Start the engine and worker (see delphos/cohort dev setup)
EVERYROW_API_URL=http://localhost:8000/api/v0 \
EVERYROW_API_KEY=<local-jwt> \
python -u scripts/test_progress_e2e.py
```

## Test File Layout

```
tests/
├── test_ops.py              # SDK operations (mocked API calls)
├── test_task.py             # Progress polling, callbacks, ETA, retries (mocked)
├── test_version.py          # Version consistency across 8 files
└── integration/             # Real API calls (gated by -m integration)
    ├── conftest.py
    ├── test_agent_map.py
    ├── test_screen.py
    ├── test_rank.py
    ├── test_dedupe.py
    ├── test_merge.py
    └── test_single_agent.py

everyrow-mcp/tests/
├── test_server.py           # MCP server tools (all mocked)
├── test_utils.py            # CSV utilities
├── test_integration.py      # MCP integration (gated by RUN_INTEGRATION_TESTS)
├── test_statusline.sh       # Status line rendering
├── test_hook_stop_guard.sh  # Stop guard hook
└── test_hook_results.sh     # Results hook + notification

scripts/                     # Manual verification (not pytest)
├── test_progress_e2e.py
└── test_progress_verbose.py
```
