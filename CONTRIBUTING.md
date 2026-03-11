# Repository Guidelines

## Project Structure & Module Organization
Core package code lives in `mesa/`. `mesa/MESA.py` contains the main implementation and exported classes such as `MESA_modality`, `MESA`, and `MESA_CV`; `mesa/__init__.py` re-exports that public API. Packaging metadata is in `setup.py`. User-facing documentation is in `README.md`, and `demo.ipynb` is the main runnable example. GitHub release publishing is defined in `.github/workflows/python-publish.yml`.

## Build, Test, and Development Commands
Use a local virtual environment before installing dependencies.

- `python -m pip install -e .` installs the package in editable mode for local development.
- `python -m pip install build` installs the build backend used by the release workflow.
- `python -m build` creates source and wheel distributions in `dist/`.
- `python -m pip install .` performs a clean install test from the current checkout.
- `python -c "from mesa import MESA_modality, MESA, MESA_CV"` is a quick smoke test for import regressions.

## Coding Style & Naming Conventions
Follow the existing Python style in `mesa/MESA.py`: 4-space indentation, module-level imports, and docstrings for public classes and functions. Preserve the current public API names, including capitalized class names and the `MESA.py` module filename, because they are part of the package interface. Prefer descriptive snake_case for functions, local variables, and new helper classes. Keep dependencies limited to the scientific Python stack already declared in `setup.py`.

## Testing Guidelines
There is no dedicated `tests/` suite yet, so validate changes with focused smoke tests and small reproducible scripts. For model pipeline changes, verify `fit`, `predict`, and `predict_proba` on a minimal pandas DataFrame and confirm the README example still imports cleanly. If you add behavior that is hard to validate manually, add a small test module under a new `tests/` directory and keep fixtures lightweight.

## Commit & Pull Request Guidelines
Recent history uses short messages such as `update` and `Update setup.py`; prefer clearer imperative subjects like `Add missing-value guard` or `Document build workflow`. Keep commits scoped to one change. Pull requests should summarize the user-visible impact, note any API or dependency changes, and include the exact commands used for validation. Attach notebook screenshots only when output formatting or plots changed.
