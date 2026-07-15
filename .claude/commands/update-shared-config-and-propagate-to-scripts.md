---
name: update-shared-config-and-propagate-to-scripts
description: Workflow command scaffold for update-shared-config-and-propagate-to-scripts in HAD-MEEG.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /update-shared-config-and-propagate-to-scripts

Use this workflow when working on **update-shared-config-and-propagate-to-scripts** in `HAD-MEEG`.

## Goal

Synchronize configuration changes (such as resource paths or environment variables) across the main config file and all scripts that consume those configs, ensuring scripts work from any working directory and are environment-driven.

## Common Files

- `src/config.py`
- `.env_example`
- `scripts/step-*.py`
- `pyproject.toml`
- `src/__init__.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Update src/config.py to add or modify config variables (e.g., resource roots, environment variable support).
- Update .env_example to document or remove relevant environment variables.
- Update scripts/step-*.py files to use the new/updated config variables instead of hardcoded paths or old config fields.
- Update pyproject.toml or src/__init__.py if project structure or exports change.
- Clean up unused imports or code left from previous config patterns.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.