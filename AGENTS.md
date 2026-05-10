# AGENTS.md

This repository is a small Python CLI. Read [`README.md`](README.md) for user-facing behavior, [`ARCHITECTURE.md`](ARCHITECTURE.md) for structure, and [`CONTRIBUTING.md`](CONTRIBUTING.md) for the human contribution workflow.

## Verify First

```bash
uv sync
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m davinci_resolve_checker --help
uvx --with tox-uv tox -e py313,lint,audit
```

## Repo Rules

- Keep host inspection in `src/davinci_resolve_checker/probes/`.
- Keep compatibility logic in `src/davinci_resolve_checker/checks/` pure over `SystemState`.
- Keep output formatting in `src/davinci_resolve_checker/render.py`.
- Extend typed models in `src/davinci_resolve_checker/models.py` instead of introducing loose dict contracts.
- Add or update focused pytest coverage when changing probe parsing, rule logic, CLI flags, or output shape.
- Prefer fixtures in `tests/conftest.py` over tests that depend on the real machine.
- Do not commit generated review artifacts such as `agent-ready-review-report.md` unless explicitly asked.

## Hand-off Rules

- Follow the branch naming, commit prefix, and PR checklist in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Before opening a PR or handing work back, run the verification commands above and report the results.

## AI Attribution

- For substantial AI-assisted commits or PRs, add an `Assisted-by:` trailer.
- For substantial AI-generated source snippets, add a `Generated-by:` source comment when appropriate under Red Hat guidance.
