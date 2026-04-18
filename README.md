# davinci-resolve-checker

Check system configuration and hardware compatibility for running DaVinci Resolve on Arch-based Linux distributions.

Supports Arch Linux, CachyOS, EndeavourOS, Manjaro, and Garuda.

## Installation

```bash
# Install uv first on Arch-based systems
sudo pacman -S uv

# Then install the checker from PyPI
uv tool install davinci-resolve-checker

# Or run it without installing
uvx davinci-resolve-checker
```

If you do not want to install `uv` from `pacman`, see the official
[uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

For the most complete diagnostics, make sure these system utilities are available:

```bash
sudo pacman -S expac clinfo mesa-utils pciutils
```

## Usage

```bash
# Run all checks
davinci-resolve-checker

# Check AMD Pro stack compatibility
davinci-resolve-checker --pro

# Machine-readable JSON output
davinci-resolve-checker --json

# Stop after the first failing result
davinci-resolve-checker --fail-fast
```

### Flags

| Flag | Description |
|------|-------------|
| `--pro` | Check for AMD proprietary stack compatibility |
| `--fail-fast` | Stop after the first failing result |
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

This project is a ground-up rewrite of [Ashark/davinci-resolve-checker](https://github.com/Ashark/davinci-resolve-checker). The check logic, GPU codename lists, and OpenCL detection heuristics are derived from the original work.

## License

[GPL-3.0](LICENSE)
