## Unreleased

### Breaking changes

- `value_counts` now behaves like pandas' `value_counts`: results are ordered most-common-first (ties by first appearance) instead of sorted by key. Items now only need to be hashable rather than orderable, and counting is O(n) rather than O(n log n).
- `tee` now raises `ValueError` when `n < 1`, as its docstring has always stated (previously `tee(0)` silently discarded the iterator and returned an empty tuple).
- `repeat(1)` now returns a new `Itr` and leaves the original exhausted, consistent with every other value of `n` (previously it returned `self`, so consuming the result also consumed the original in that case only).

### New features

- `next_if(predicate)`: consume and return the next item only if it satisfies the predicate, otherwise leave it in place and return `None` (like Rust's `Peekable::next_if`).
- `sum()` / `prod()`: terminal aggregations over the remaining items.
- `dedup()`: lazily remove *consecutive* duplicates, keeping the first of each run (like Rust's `dedup`); works on infinite iterators.
- `zip_longest(other, fillvalue=...)`: like `zip`, but continues to the end of the longer input, padding with `fillvalue`.
- `sorted_by(key, reverse=...)`: eager stable sort by a key function.

### Bug fixes

- Removed the broken `apidoc` console script from the distribution: the wheel does not package the `scripts` module, so the installed command always failed with `ModuleNotFoundError`. It is a dev-only tool, now run directly from the source tree.

## 0.3.0

### Breaking changes

- `nth` is now **0-based**, consistent with Rust's `Iterator::nth` and Python's indexing conventions: `nth(0)` returns the first item (previously this raised `ValueError` and `nth(1)` returned the first item). Update callers by dropping the `+ 1`.
- `interleave` now yields the remaining elements of the longer iterable once the shorter one is exhausted, matching Rust's `interleave` (previously it stopped at the shorter input, silently dropping the tail).

### New features

- `chunk_by`: lazily group *consecutive* elements sharing a key (the semantics of `itertools.groupby` / Rust's `chunk_by`). Unlike `groupby` it does not sort, so it preserves order and works on infinite iterators.

### Documentation

- Clarified that `groupby` (and `value_counts`, which builds on it) is **eager**: it sorts the entire input up front, so it reorders output, requires mutually-orderable keys, and must not be used on infinite sources. Corrected the lazy/eager categorisation in the README.
- Corrected the `nth` docstring, which previously claimed it returned `None` when out of range (it raises `StopIteration`).
