## Unreleased

### Breaking changes

- Python 3.12 is no longer supported. Supported versions are now 3.13, 3.14 and 3.15.

## 0.5.0

### Breaking changes

- Terminal methods no longer raise `StopIteration` when an item is missing. A `StopIteration` escaping a `map`/`filter` callback silently ends the *enclosing* iteration, so e.g. `Itr(groups).map(lambda g: Itr(g).reduce(max))` used to drop every group after the first empty one without any error. Now:
  - `position` returns `None` when nothing matches (like `find`, and Rust's `Option`).
  - `nth` raises `IndexError` when the iterator is too short, like indexing a sequence past its end (a negative `n` is still a `ValueError`).
  - `reduce` raises `TypeError` on an empty iterator, like `functools.reduce`.
  - `peek` raises `ValueError` on an exhausted iterator, consistent with `last`, `max` and `min`.
  - `next()` still raises `StopIteration`, as the builtin does.
- `partition` calls the predicate once per item rather than twice, buffering items for whichever side is not being read. Side-effecting predicates now see each item exactly once.

### New features

- `filter_map(f)` / `find_map(f)`: map and drop `None` results / return the first non-`None` result (Rust's `filter_map` and `find_map`). `filter_map` is typed `Itr[U]` rather than `Itr[U | None]`.
- `unique(key=None)`: lazy, order-preserving removal of duplicates anywhere in the input (not just adjacent ones, unlike `dedup`). Items or keys must be hashable.
- `min_max(key=None)`: the minimum and maximum in a single pass, with the builtins' tie-breaking.
- `compress(selectors)`: keep the items whose selector is truthy (`itertools.compress`).
- `take_last(n)`: the last `n` items, holding at most `n` in memory (eager, like `rev`).
- `chain(*others)` and `interleave(*others)` accept any number of iterables; `interleave` round-robins across them.
- `zip(other, *, strict=False)`: pass `strict=True` to raise `ValueError` on a length mismatch.
- `groupby(key, *, sort=True)`: `sort=False` emits groups in order of first appearance, and needs only hashable (not orderable) keys.
- `sorted_by(key=None, ...)`: the key is now optional, sorting by the items themselves.
- `all()` / `any()`: the predicate is now optional, defaulting to truthiness.
- `sum(start=0)` / `prod(start=1)`: optional start value, e.g. `sum([])` to concatenate lists.
- `Itr` now has a `repr` showing the wrapped iterator (without consuming it).

- Installable **agent skill**: the package now bundles a `SKILL.md` reference for AI coding agents, plus an `itrx-skill` console script to symlink it into a project (`itrx-skill --install [PATH]` / `--remove [PATH]`, default `PATH=.agents`, creating `PATH/skills/itrx`). The symlink points at the skill inside the installed `itrx`, so it always matches the version in use. See the "Agent skill" section of the README.
- `dedup_with_count()`: the lazy, positional counterpart to `value_counts` — collapses each *consecutive* run of equal items into an `(item, count)` pair (run-length encoding), preserving order and working on infinite iterators. Note the `(item, count)` ordering matches `value_counts` and is the reverse of Rust's `dedup_with_count`.
- `scan(init, func)`: lazily map items through a running state, optionally stopping early (Rust's `Iterator::scan`). Generalises `accumulate`: the state need not share the items' type, and returning `None` halts iteration. Returning `(new_state, None)` still yields `None` as an output, so the halt signal is never ambiguous.
- `is_sorted(key=None, *, reverse=False)`: check whether the remaining items are in order, short-circuiting at the first inversion. Non-strict, so equal runs count as sorted. Covers Rust's `is_sorted` and `is_sorted_by_key`, and adds `reverse` for symmetry with `sorted_by`.
- `eq(other)`: element-wise comparison against another iterable, short-circuiting at the first difference instead of materialising both sides (Rust's `Iterator::eq`). Note `Itr` defines no `__eq__`, so `==` remains an identity check.

## 0.4.0

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
