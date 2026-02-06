# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/lerobot/`, with domain modules such as `datasets/`, `policies/`, `envs/`, `robots/`, `teleoperators/`, and `utils/`. Configuration and runnable entrypoints are under `src/lerobot/configs/` and `src/lerobot/scripts/`. Tests live in `tests/` and largely mirror the package layout; large fixtures are stored in `tests/artifacts/`. Supporting material includes `docs/` (documentation), `examples/` (usage demos), `benchmarks/`, `docker/`, and `media/`.

## Build, Test, and Development Commands
- `poetry sync --extras "dev test"` or `uv sync --extra dev --extra test`: install dev/test dependencies.
- `poetry sync --all-extras` or `uv sync --all-extras`: include optional simulation environments for full test parity.
- `pip install -e .`: editable install when using pip.
- `python -m pytest -sv ./tests`: run the full test suite.
- `pytest tests/<TEST_TO_RUN>.py`: run a focused test file.
- `pre-commit install` then `pre-commit`: enable and run formatting/lint hooks.
- `make build-user` / `make build-internal`: build Docker images.
- `make test-end-to-end DEVICE=cpu`: optional end-to-end training/eval smoke tests (writes to `tests/outputs/`).

## Coding Style & Naming Conventions
Use `ruff` via `pre-commit` for formatting and linting (see `pyproject.toml`). Stick to 4-space indentation and PEP 8 naming: `snake_case` for functions/variables, `CapWords` for classes, and `SCREAMING_SNAKE_CASE` for constants. Add new modules under the closest domain package (for example, a new policy goes in `src/lerobot/policies/`).

## Testing Guidelines
Tests use `pytest` (with `pytest-cov` available). Name test files `test_*.py` and test functions `test_*`. Keep tests close to the module they cover inside `tests/`. If fixtures are missing, install Git LFS and run `git lfs pull` to populate `tests/artifacts/`. No explicit coverage threshold is documented; add targeted tests for new behavior and run impacted suites before opening a PR.

## Commit & Pull Request Guidelines
Recent commits are short, imperative summaries; many follow Conventional Commits with scopes (for example, `fix(docs): ...`) and often include PR numbers like `(#1234)`. Use clear, descriptive messages in that style. PRs should have a summary title, link related issues in the description, mark work-in-progress as draft or prefix `[WIP]`, and ensure tests pass. Include screenshots or extra context when changes affect docs or user-facing behavior.
