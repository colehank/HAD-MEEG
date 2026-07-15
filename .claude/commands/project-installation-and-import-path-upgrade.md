---
name: project-installation-and-import-path-upgrade
description: Workflow command scaffold for project-installation-and-import-path-upgrade in HAD-MEEG.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /project-installation-and-import-path-upgrade

Use this workflow when working on **project-installation-and-import-path-upgrade** in `HAD-MEEG`.

## Goal

Upgrade the Python project to be installable/editable (e.g., via hatchling/uv), so scripts can import from src without sys.path hacks, and regenerate lockfiles accordingly.

## Common Files

- `pyproject.toml`
- `scripts/step-*.py`
- `uv.lock`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Update pyproject.toml to add or change build backend and package config.
- Remove sys.path manipulation boilerplate from all scripts.
- Regenerate uv.lock to reflect new editable install and dependencies.
- Test that scripts can import from src without sys.path hacks.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.