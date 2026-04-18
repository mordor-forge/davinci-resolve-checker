# davinci-resolve-checker

Check system configuration and hardware compatibility for running DaVinci Resolve on Arch-based Linux distributions.

Supports Arch Linux, CachyOS, EndeavourOS, Manjaro, and Garuda.

## Installation

```bash
uv tool install davinci-resolve-checker
# or
pipx install davinci-resolve-checker
```

## Usage

```bash
# Run all checks
davinci-resolve-checker

# Check AMD Pro stack compatibility
davinci-resolve-checker --pro

# Machine-readable JSON output
davinci-resolve-checker --json

# Stop after first failure
davinci-resolve-checker --fail-fast
```

### Flags

| Flag | Description |
|------|-------------|
| `--pro` | Check for AMD proprietary stack compatibility |
| `--fail-fast` | Stop after first failure |
| `--json` | Machine-readable JSON output |

### Shell Completions

```bash
# Install completions for your current shell (bash, zsh, or fish)
davinci-resolve-checker --install-completion
```

This auto-detects your shell and adds the completion script to your RC file. Restart your shell or source the RC file to activate.

### Exit Codes

- `0` — all checks pass
- `1` — at least one check failed

## Development

```bash
git clone https://github.com/mordor-forge/davinci-resolve-checker.git
cd davinci-resolve-checker
uv sync
uv run pytest -v
```

### Run the full test matrix

```bash
uvx --with tox-uv tox
```

### Lint

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Credits

This project is a ground-up rewrite of [Ashark/davinci-resolve-checker](https://github.com/Ashark/davinci-resolve-checker). The check logic, GPU codename lists, and OpenCL detection heuristics are derived from the original work. No code was copied verbatim.

## License

GPL-3.0 — inheriting the copyleft license from the original project.

Copyright (c) 2024 Ashark (original project)
Copyright (c) 2026 mordor-forge (rewrite)
