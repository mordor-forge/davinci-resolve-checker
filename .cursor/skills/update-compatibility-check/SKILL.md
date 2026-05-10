---
name: update-compatibility-check
description: Update DaVinci Resolve compatibility probes, rules, and tests safely. Use when changing GPU detection, OpenCL parsing, vendor-specific checks, CLI diagnostics, or test fixtures in this repository.
---

# Update Compatibility Check

## Use This Skill For

- adding or changing probe behavior in `src/davinci_resolve_checker/probes/`
- adjusting AMD, NVIDIA, or common compatibility rules
- updating CLI or renderer behavior that depends on `CheckResult` output

## Workflow

1. Read `ARCHITECTURE.md` and `AGENTS.md` first.
2. Choose the correct layer:
   - `probes/` for host inspection and parsing
   - `checks/` for pure compatibility decisions
   - `models.py` for shared typed contracts
   - `render.py` for presentation only
3. Update shared fixtures in `tests/conftest.py` when the system model changes.
4. Add focused pytest coverage for both the success path and the failure path.
5. Run the verification commands below before handing work back.

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m davinci_resolve_checker --help
```

## Repo-Specific Rules

- Keep checks pure over `SystemState`; do not print or shell out from `checks/`.
- Prefer mocks and fixtures in tests over depending on the real host machine.
- Treat JSON output changes as interface changes and cover them explicitly.
- Preserve `fail_fast` behavior when touching `run_all_checks()`.
