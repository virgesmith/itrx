# Agent Guidelines for `itrx`

This file instructs AI agents acting as developer, reviewer, and QA for this repository.

## Project Overview

`itrx` is a Python library providing a chainable, lazy iterator adapter (`Itr`) inspired by Rust's `Iterator` trait. The entire public API lives in [src/itrx/itr.py](src/itrx/itr.py). Tests are in [src/test/](src/test/) and doctests appear inline in the source and in [README.md](README.md).

## Toolchain

| Tool | Command |
|------|---------|
| Package manager | `uv` |
| Linter / formatter | `ruff` (`uv run ruff check`, `uv run ruff format`) |
| Type checker | `ty` (`uv run ty check src`) |
| Tests + coverage | `uv run pytest` |
| Install dev deps | `uv sync --dev` |

Pre-commit hooks run `uv-lock`, `ruff-check --fix`, `ruff-format`, and `ty` automatically on commit.

## Quality Gates

All of the following must pass before any change is considered complete:

```sh
uv run ruff check          # zero lint errors
uv run ruff format --check # zero formatting issues
uv run ty check src        # zero type errors
uv run pytest              # all tests pass, 100% coverage required
```

Coverage is enforced at 100% (`--cov-fail-under=100`). Every new code path needs a test or a doctest.

## Developer Rules

- **No new runtime dependencies.** This is a zero-dependency library. Dev-only tools go in `[dependency-groups.dev]` in [pyproject.toml](pyproject.toml).
- **Lazy by default.** New methods on `Itr` must return an `Itr` (i.e. stay lazy) unless they are terminal consumers (like `collect`, `for_each`, `fold`).
- **Doctests are first-class.** Every public method must have a doctest that serves as both documentation and test. Keep examples minimal and self-contained.
- **Type annotations required.** All function signatures need full annotations. `ty` will catch missing or incorrect ones.
- **Line length is 120** (configured in [pyproject.toml](pyproject.toml) under `[tool.ruff]`).
- **No comments explaining what the code does.** Only add a comment when the *why* is non-obvious (hidden constraint, workaround, subtle invariant).

## Reviewer Checklist

When reviewing a PR or diff, check:

1. **Correctness** — does the logic match the documented behaviour? Edge cases: empty iterables, infinite sequences, single-element, type boundaries.
2. **API consistency** — new methods should follow the naming and signature conventions of existing ones. Compare against Rust's `Iterator` trait for inspiration.
3. **Laziness** — intermediate methods must not eagerly consume the iterator.
4. **Coverage** — every branch must be exercised. No `# pragma: no cover` without a very good reason.
5. **Doctests** — run them mentally; verify the output is correct and the example is illuminating.
6. **Types** — return types and generics should be precise. Avoid `Any` unless unavoidable.
7. **Ruff rules** — no rule in the `select` list should be suppressed without justification. The active rules are: `ARG, B, C, D103, E, F, I, N, PERF, PTH, RET, RUF, SIM, UP, W` (E501 is ignored).
8. **README / apidoc** — if a public method is added or its signature changes, update [README.md](README.md) and regenerate [doc/apidoc.md](doc/apidoc.md) with `uv run apidoc`.

## QA Rules

- Run the full gate suite (`ruff check`, `ruff format --check`, `ty check`, `pytest`) before declaring any task done.
- CI runs the matrix: Python 3.12, 3.13, 3.14 × ubuntu, windows, macos. Flag anything that might be platform- or version-specific.
- If a test is skipped or marked `xfail`, leave a comment explaining why and when it can be removed.
- Coverage HTML is uploaded as a CI artefact for Linux/Python 3.13. Check it for any uncovered lines after adding new code.
- Doctests in [README.md](README.md) are executed by pytest (`--doctest-glob=README.md`). Keep them runnable.

## Repository Layout

```
src/
  itrx/
    itr.py          # entire library implementation
    __init__.py     # public exports
    py.typed        # PEP 561 marker
  test/
    test_aggregation.py
    test_collection.py
    test_combine_split.py
    test_general.py
    test_transform_filter.py
doc/
  apidoc.md         # generated — do not edit by hand
  examples.ipynb
README.md           # contains executable doctests
pyproject.toml
.pre-commit-config.yaml
.github/workflows/
  lint-test.yml     # CI: lint + type check + test matrix
  publish.yml       # CI: PyPI publish on tag
```

## Branch and Release Policy

- **`main` is branch-protected** (verified via GitHub API). Direct pushes are blocked for non-admins. All changes must go through a pull request.
- **Releases are triggered by a `v*` tag** (e.g. `v1.2.3`). Pushing such a tag to GitHub runs [publish.yml](.github/workflows/publish.yml), which builds a wheel and publishes it to PyPI using trusted publishing (OIDC — no API token needed). Do not push a `v*` tag unless the release is fully ready and the version in [pyproject.toml](pyproject.toml) matches the tag.
- Version bumps go in `pyproject.toml` (`version = "x.y.z"`). Update [relnotes.md](relnotes.md) at the same time.

## Workflow

1. Create a feature branch off `main` — never commit directly to `main`.
2. Make changes in [src/itrx/itr.py](src/itrx/itr.py).
3. Add or update tests in the matching file under [src/test/](src/test/).
4. Add or update the inline doctest on the method.
5. Run the full gate suite locally.
6. If the public API changed, update [README.md](README.md) and run `uv run apidoc`.
7. Commit — pre-commit hooks will auto-fix formatting and re-lock `uv.lock`.
8. Open a PR targeting `main`; CI must pass before merging.
9. To release: bump the version in [pyproject.toml](pyproject.toml), update [relnotes.md](relnotes.md), merge to `main`, then push a `vX.Y.Z` tag — PyPI publish triggers automatically.
