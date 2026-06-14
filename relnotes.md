## 0.3.0

### Breaking changes

- `nth` is now **0-based**, consistent with Rust's `Iterator::nth` and Python's indexing conventions: `nth(0)` returns the first item (previously this raised `ValueError` and `nth(1)` returned the first item). Update callers by dropping the `+ 1`.
- `interleave` now yields the remaining elements of the longer iterable once the shorter one is exhausted, matching Rust's `interleave` (previously it stopped at the shorter input, silently dropping the tail).

### New features

- `chunk_by`: lazily group *consecutive* elements sharing a key (the semantics of `itertools.groupby` / Rust's `chunk_by`). Unlike `groupby` it does not sort, so it preserves order and works on infinite iterators.

### Documentation

- Clarified that `groupby` (and `value_counts`, which builds on it) is **eager**: it sorts the entire input up front, so it reorders output, requires mutually-orderable keys, and must not be used on infinite sources. Corrected the lazy/eager categorisation in the README.
- Corrected the `nth` docstring, which previously claimed it returned `None` when out of range (it raises `StopIteration`).
