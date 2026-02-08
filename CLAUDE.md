# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A time-series stats collector for the [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer) GitHub project. It periodically snapshots repo metrics (issues by label, stars, forks, languages, etc.) and appends them as newline-delimited JSON to files in `stats/`. A GitHub Actions workflow runs every 8 hours and auto-commits the results.

## Commands

```bash
make deps          # Install uv (if needed) and sync Python dependencies (into .venv/)
make clean         # Remove uv, virtualenv, and lockfile

# Run the stats collector (requires GITHUB_TOKEN env var or --access-token):
uv run python main.py stats stats/compiler-explorer.json

# Generate graphs from collected stats (no token needed):
uv run python main.py graph stats/compiler-explorer.json graphs/
```

## Architecture

This is a single-file Python CLI (`main.py`) using Click and PyGithub:
- `cli()` - Click group; sets up GitHub client (lazily) from `--access-token` or `GITHUB_TOKEN` env var
- `stats()` - Fetches repo metadata (issues by label/state, languages, stars, forks, watchers, head revision) and appends one JSON object per line to the output file
- `graph()` - Reads NDJSON stats file and generates time-series graphs to an output directory (no GitHub token needed)

The stats JSON files are append-only (one JSON object per line, not a JSON array). Each line is a complete snapshot.

## Code Style

- Linter/formatter: ruff (120-char line length, targeting Python 3.10)
- Python version: >=3.9
- Dependencies managed via uv (virtualenv lives in `.venv/`)
- Pre-commit hooks: `pre-commit install` (requires globally-installed `pre-commit`)
