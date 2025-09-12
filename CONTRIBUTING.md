# Contributing to MLQuantumSwitch

Thanks for your interest in contributing! This repo contains utilities for quantum measurement incompatibility and clustering. Small fixes, docs, and features are welcome.

## Getting started
1. Use Python 3.10+ (3.12 works).
2. Create a virtual environment and install deps:
   ```powershell
   python -m venv .venv ; .\.venv\Scripts\Activate.ps1
   pip install -U pip
   pip install -r requirements.txt
   ```
3. Optional: install `hdbscan` if you plan to use HDBSCAN clustering.

## Development workflow
- Create a branch from `main`:
  - `feature/<short-name>` for new features
  - `fix/<short-name>` for bug fixes
  - `docs/<short-name>` for documentation only
- Keep changes focused and add/update docstrings.
- Add small unit checks or notebook snippets if behavior changes.

## Commit messages
Use concise, descriptive messages. Conventional prefixes help:
- `feat:` new functionality
- `fix:` bug fix
- `docs:` docs/readme changes
- `chore:` tooling or maintenance

Example: `feat: add k-medoids linear++ init option`

## Code style
- Follow PEP 8 and prefer type hints.
- Keep functions stateless and pure where possible.
- Avoid unnecessary heavy dependencies; use NumPy/QuTiP primitives.

## Pull requests
- Rebase on top of `main` before opening the PR.
- Describe the goal and key changes; add before/after examples if relevant.
- Ensure the notebook still runs (if you modified workflows).

## License
By contributing, you agree that your contributions will be licensed under the MIT License in `LICENSE`.
