# Repository Guidelines

This guide helps contributors and agents work effectively in this repo.

## Project Structure & Module Organization
- Source: `lobster/`
  - `agents/` (agent logic, supervisors), `tools/` (analysis services), `core/` (data, backends, schemas, clients), `config/` (settings, registry), `ui/` (CLI/Streamlit).
- Tests: `tests/` with `unit/`, `integration/`, `system/`, `performance/` and `mock_data/`.
- Docs: `docs/`; Examples: `examples/`; CI: `.github/workflows/`; Docker: `Dockerfile`, `docker/`.

## Build, Test, and Development Commands
- Setup: `make dev-install` (venv, deps, pre-commit, `.env`), `make setup-env` (create `.env`).
- Run locally: `make run` or `lobster chat`; module entry: `python -m lobster`.
- Tests: `make test` (coverage), `make test-fast` (parallel), `make test-integration`.
- Quality: `make format` (black+isort), `make lint` (flake8+pylint+bandit), `make type-check` (mypy).
- Docker: `make docker-build`, `make docker-run`.

## Coding Style & Naming Conventions
- Python 3.12+, 4-space indent, max line length 88.
- Format with black; sort imports with isort (profile=black).
- Lint with flake8 (Bugbear, etc.) and pylint; security with bandit.
- Type hints required; keep functions small and testable.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.

## Testing Guidelines
- Framework: pytest; discovery `tests/test_*.py`.
- Markers available (e.g., `@pytest.mark.unit`, `integration`, `performance`).
- Coverage: threshold 80% (see `pytest.ini`); generate `htmlcov/` with `make test`.
- Example: `pytest tests/unit/tools/test_gpu_detector.py -v -m unit`.

## Commit & Pull Request Guidelines
- Commits: prefix with intent (e.g., `Add:`, `Fix:`, `Refactor:`, `Docs:`). Keep messages imperative and scoped.
- Branches: `feature/<slug>`, `fix/<slug>`.
- PRs: include clear description, linked issues, tests, and updates to docs/examples as needed. Attach screenshots for UI changes.

## Security & Configuration Tips
- Do not commit secrets. Use `.env` (see `.env.example`); required keys include `OPENAI_API_KEY`, optional AWS Bedrock keys.
- Pre-commit hooks (`pre-commit install`) enforce formatting, linting, and secret scanning.

## Architecture Overview
- UI: CLI (`lobster/cli.py` → `lobster chat`) and Streamlit (`lobster/streamlit_app.py`).
- Smart client: `LOBSTER_CLOUD_KEY` selects Cloud client vs local `AgentClient`; both implement `core/interfaces/base_client.py` for a unified API.
- Local pipeline: Agents in `lobster/agents/` orchestrate stateless services in `lobster/tools/`; `core/data_manager_v2.py` manages modalities (AnnData), provenance, and errors.
- Services (stateless): preprocessing, QC, clustering/UMAP, GEO download, pseudobulk, differential formula construction, visualization, plus proteomics preprocessing/quality/differential/plots.
- Data layer: adapters in `core/adapters/`, schemas in `core/schemas/`, storage backends `core/backends/h5ad_backend.py` and `core/backends/mudata_backend.py`.
- Registry: `config/agent_registry.py` and `config/agent_config.py` provide a single source of truth for agent availability and tool wiring.
