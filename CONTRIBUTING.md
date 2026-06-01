# Contributing

Hello human and AI contributors, this document exists to help you understand the project and set some rules for contributions.

## Contribution Workflow

Before opening a pull request, search existing issues and PRs to avoid duplicate work.
Keep each PR focused on one feature, bug fix, or vendor update.
Avoid mixing unrelated Python changes, generated binding updates, documentation edits, and `vendor/llama.cpp` changes unless they are required for the same fix.

Describe what changed, why it changed, and how it was tested.
Link relevant issues, include any required build flags or hardware assumptions, and add a `CHANGELOG.md` entry for user-visible fixes or features (see `CHANGELOG.md` for examples).

BREAKING CHANGES WILL ALMOST CERTAINLY BE REJECTED OR REFACTORED.

## PR Titles and Changelog Entries

Use PR titles in the form `<tag>: <title>`, with an optional scope when it adds clarity: `feat: add X`, `fix(server): handle Y`, `fix(ci): repair Z`, or `chore: bump version to N`.
Prefer tags already used in the project history, such as `feat`, `fix`, `chore`, `ci`, `docs`, and `refactor`.

Add changelog entries under `## [Unreleased]` using the PR title followed by `by @contributor in #1234`.

```md
- feat(server): add support for X by @contributor in #1234
- fix(ci): repair Y wheel builds by @contributor in #1234
```

## Local Development

Prerequisites: Python 3.8+, CMake 3.21+, a C/C++ compiler, and Git submodules.
From a fresh checkout of the repository, initialize submodules and create a virtual environment:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make deps
make build
```

Run tests and lint checks before submitting changes:

```bash
make test
make lint
```

Use backend-specific build targets when validating native acceleration or backend-specific fixes, for example `make build.openblas`, `make build.cuda`, `make build.metal`, or `make build.vulkan`.

## Testing Expectations

Add or update tests for behavior changes or fixing regressions.
The test suite uses pytest and lives under `tests/`; name files `test_*.py` and test functions `test_*`.

For changes involving native backends, model behavior, performance, or platform compatibility, document the environment used for validation in the PR.
If a change cannot be covered by automated tests, include a short manual validation recipe instead.

## Code Style

Python code is formatted with Ruff using an 88-character line length.
Run `make format` to apply automatic fixes and `make lint` to check formatting and lint rules.

Use 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and `UPPER_CASE` for constants.
Follow existing patterns when touching ctypes bindings or server APIs, and avoid adding dependencies unless they are necessary for the feature or fix.

## Documentation Style

Write Markdown with one sentence or core idea per physical line to keep diffs focused and easier to review.
Do not manually wrap lines at a fixed column width.
Keep `README.md` focused on user-facing setup and usage; link to this guide for contribution workflow details rather than duplicating them.

## Project Layout

The Python package lives in `llama_cpp/`, with tests in `tests/` and examples in `examples/`.
Documentation lives in `docs/` and is built with `mkdocs.yml`.
The `vendor/llama.cpp/` directory is a Git submodule containing the upstream llama.cpp sources used by the bindings.
