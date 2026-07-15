```markdown
# HAD-MEEG Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill covers the core development patterns and workflows used in the HAD-MEEG Python codebase. You'll learn the project's coding conventions, how to update shared configuration, modernize project installation, and the typical approaches to testing and script organization. This guide is ideal for contributors aiming to maintain consistency and efficiency in the HAD-MEEG repository.

## Coding Conventions

**File Naming**

- Use kebab-case for file names.
  - Example: `data-loader.py`, `step-preprocess.py`

**Import Style**

- Prefer relative imports within the `src` package.
  - Example:
    ```python
    from .config import DATA_ROOT
    ```

**Export Style**

- Use named exports (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ["load_data", "process_signal"]
    ```

**Commit Patterns**

- Commit messages are freeform, with no enforced prefix.
- Average commit message length: ~56 characters.

## Workflows

### Update Shared Config and Propagate to Scripts

**Trigger:** When you need to change how scripts locate resources or read configuration (e.g., switching from hardcoded paths to environment variables).

**Command:** `/sync-config-to-scripts`

1. Update `src/config.py` to add or modify configuration variables (such as resource roots or environment variable support).
    ```python
    # src/config.py
    import os

    DATA_ROOT = os.getenv("DATA_ROOT", "data/")
    ```
2. Update `.env_example` to document or remove relevant environment variables.
    ```
    # .env_example
    DATA_ROOT=./data/
    ```
3. Update all `scripts/step-*.py` files to use the new or updated config variables instead of hardcoded paths.
    ```python
    # scripts/step-preprocess.py
    from src.config import DATA_ROOT
    ```
4. Update `pyproject.toml` or `src/__init__.py` if the project structure or exports change.
5. Clean up unused imports or code left from previous config patterns.

**Files involved:**  
- `src/config.py`  
- `.env_example`  
- `scripts/step-*.py`  
- `pyproject.toml`  
- `src/__init__.py`

---

### Project Installation and Import Path Upgrade

**Trigger:** When you want to modernize the project structure for better dependency management and import hygiene.

**Command:** `/upgrade-project-install`

1. Update `pyproject.toml` to add or change the build backend and package configuration.
    ```toml
    # pyproject.toml
    [build-system]
    requires = ["hatchling"]
    build-backend = "hatchling.build"
    ```
2. Remove `sys.path` manipulation boilerplate from all scripts.
    ```python
    # Before
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
    from config import DATA_ROOT

    # After
    from src.config import DATA_ROOT
    ```
3. Regenerate `uv.lock` to reflect the new editable install and dependencies.
    ```
    uv pip install -e .
    uv pip freeze > uv.lock
    ```
4. Test that scripts can import from `src` without any `sys.path` hacks.

**Files involved:**  
- `pyproject.toml`  
- `scripts/step-*.py`  
- `uv.lock`

---

## Testing Patterns

- Test files follow the pattern `*.test.*` (e.g., `data-loader.test.py`).
- The specific testing framework is not detected; check individual test files for details.
- To run tests, use your preferred Python test runner (e.g., `pytest`), or refer to project documentation if available.

## Commands

| Command                   | Purpose                                                                    |
|---------------------------|----------------------------------------------------------------------------|
| /sync-config-to-scripts   | Synchronize config changes across config files and scripts                 |
| /upgrade-project-install  | Upgrade project to editable install and clean up import paths              |
```
