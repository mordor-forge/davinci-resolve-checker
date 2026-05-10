# Contributing

Thanks for helping improve `davinci-resolve-checker`.

This project is a small Python CLI, so the main goal is to keep changes easy to review, easy to test, and easy to trust on real Linux systems.

## Report Bugs

Open a GitHub issue with:

- your distribution and version
- GPU model and driver in use
- the command you ran
- the checker output, preferably with `--json` if possible
- what you expected instead

If the bug depends on local packages or drivers, include the relevant package versions and whether `glxinfo`, `eglinfo`, and `clinfo` are available.

## Request Features

Open a GitHub issue describing:

- the user problem you are trying to solve
- the hardware or distro scenario involved
- whether the change adds a new probe, a new compatibility rule, or a new CLI behavior

Small, focused proposals are easier to review than broad rewrites.

## Security Reporting

Do not open a public issue for a vulnerability that could put users at risk.

Instead, contact the maintainers privately through GitHub security reporting or another private maintainer channel before public disclosure. Public issues are fine for hardening ideas that do not expose an active security problem.

## Development Setup

```bash
git clone git@github.com:mordor-forge/davinci-resolve-checker.git
cd davinci-resolve-checker
uv sync
uv run pytest -v
```

Useful local checks:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uvx --with tox-uv tox
uv run python -m davinci_resolve_checker --help
```

## Branching Strategy

- Branch from `main`
- Keep each branch scoped to one change or one tightly related set of changes
- Prefer descriptive branch names such as `fix/nvidia-opencl-check`, `docs/architecture`, or `chore/agent-readiness-docs`

## Coding Standards

- Keep system probing logic in `src/davinci_resolve_checker/probes/`
- Keep compatibility rules in `src/davinci_resolve_checker/checks/`
- Keep rendering and output formatting in `src/davinci_resolve_checker/render.py`
- Extend `SystemState`, `CheckResult`, and related models instead of passing ad hoc dictionaries around
- Preserve machine-readable JSON output shape unless the change intentionally updates the interface

## Testing Expectations

- Add or update focused pytest coverage for any behavior change
- Prefer fixture-driven tests over shelling out in tests
- When changing probe parsing, include success and failure-path coverage
- When changing CLI behavior or output shape, add a targeted CLI or renderer test
- The repository currently targets full coverage in CI, so new logic should ship with tests

## Commit Messages

Use the existing commit style from the repository history:

- `fix: ...`
- `docs: ...`
- `ci: ...`
- `chore: ...`

Make the subject describe the user-visible reason for the change, not just the file you touched.

## Pull Requests

- Open the PR against `main`
- Summarize the problem and the chosen fix
- List the verification you ran locally
- Keep generated artifacts out of the PR unless they are intentionally part of the change
- Wait for CI to pass before asking for final review

If a change affects real hardware behavior, call out the expected GPU and driver scenarios in the PR description.
