---
name: itrx
description: >
  Use when writing, editing, or debugging code that uses itrx's `Itr` — a chainable, lazy
  iterator adaptor for Python inspired by Rust's Iterator trait. Covers the lazy/eager split,
  the full method set by category, single-pass and infinite-iterator pitfalls, and how
  `collect` materialises results. Triggers: "itrx", "Itr", chainable/fluent iterator, a
  Rust-style `.map().filter().collect()` chain in Python, replacing nested `itertools` calls.
---

# Developing with itrx

`itrx` wraps any Python iterable, iterator, or generator in an `Itr[T]` that exposes a fluent,
chainable, mostly-lazy API modelled on Rust's [`Iterator`](https://doc.rust-lang.org/std/iter/trait.Iterator.html)
trait. It is zero-dependency: in most cases each method is thin syntactic sugar over `itertools`
or a builtin, so the runtime cost over hand-written `itertools` is negligible.

```py
from itrx import Itr

Itr(range(100)).rev().step_by(4).skip(10).map(lambda x: x * x).filter(lambda x: x % 10 == 9).for_each(print)
```

`Itr` implements the iterator protocol itself (`__iter__`/`__next__`), so an `Itr` can be passed
anywhere an iterator is accepted — `for x in itr`, `list(itr)`, `sum(itr)`, or as the `other`
argument to `zip`/`chain` on another `Itr`.

This skill is a quick reference, not the full API. For a method's exact signature, parameters,
raises and doctest, consult the generated per-method reference:

<https://github.com/virgesmith/itrx/blob/main/doc/apidoc.md>

Read it whenever you need a detail this file doesn't state — it is generated from the source
docstrings and covers every public method. Note it tracks `main`, so it may describe methods
newer than the installed version; `Itr.<method>.__doc__` (or `help(Itr.<method>)`) is the
authoritative check for what the *installed* version actually has. In a checkout of the itrx repo
itself the same file is at `doc/apidoc.md`, and the narrative docs are in `README.md`.

## When reaching for `Itr` is worth it

The reason to use it is **readability**, not speed — it will not make a pipeline faster than the
equivalent `itertools` code, because it *is* that code underneath. Reach for it when:

- **A pipeline reads inside-out as builtins.** `filter(lambda .., map(lambda .., islice(islice(reversed(x), None, None, 4), 10, None)))`
  has to be read from the innermost call outwards; the `Itr` chain reads left to right in the
  order the data actually flows.
- **You want laziness by default without writing generators.** Every intermediate step returns a
  new `Itr` and pulls items only on demand, so an infinite source stays workable right up to the
  terminal call.
- **The operation exists in Rust's `Iterator` but not as a Python builtin** — `fold`, `inspect`,
  `partition`, `position`, `intersperse`, `interleave`, `dedup`, `dedup_with_count`, `chunk_by`,
  `unzip`, `map_while`, `scan`, `is_sorted`, `eq`, `next_chunk`, `step_by`, `rolling`.

Conversely, it is **not** worth it for a single `map`/`filter` (a comprehension is clearer), for
code already dominated by numpy/pandas vectorised calls, or where the data is a materialised
`list` you are indexing into rather than streaming.

## Constructing

`Itr(it)` accepts any `Iterable[T]` and immediately calls `iter()` on it. Wrapping a **sequence**
(list, tuple, str, range) is safe to do repeatedly — each `Itr` gets its own fresh iterator — but
wrapping a **generator or iterator** hands over ownership: consuming the `Itr` consumes it.

## Lazy vs eager — the single most important distinction

**Lazy** (return a new `Itr` and consume nothing until pulled, so they are safe on infinite
sources — with the exception of `product`, which materialises `other` up front like
`itertools.product` does):

`accumulate`, `batched`, `chain`, `chunk_by`, `copy`, `cycle`, `dedup`, `dedup_with_count`,
`enumerate`, `filter`,
`flat_map`, `flatten`, `inspect`, `interleave`, `intersperse`, `map`, `map_dict`, `map_while`,
`pairwise`, `partition`, `product`, `repeat`, `rolling`, `scan`, `skip`, `skip_while`,
`step_by`, `take`,
`take_while`, `tee`, `unzip`, `zip`, `zip_longest`

**Eager** (consume the iterator, return a concrete value; **never** on an infinite source):

- Collection: `collect`, `last`, `next`, `next_chunk`, `next_if`, `nth`, `position`, `peek`
- Aggregation: `all`, `any`, `consume`, `count`, `eq`, `find`, `fold`, `for_each`, `is_sorted`,
  `max`, `min`, `prod`, `reduce`, `sum`
- Whole-input reordering: `groupby`, `sorted_by`, `value_counts`, `rev`

Note that some of these consume only as far as they need to: `next`, `next_chunk`, `nth`,
`next_if`, `peek`, `find`, `position`, `any` and `all` short-circuit, so they *are* safe on an
infinite source. `collect`, `count`, `last`, `consume`, `fold`, `reduce`, `sum`, `prod`, `max`,
`min`, `for_each`, `rev`, `groupby`, `sorted_by` and `value_counts` are not. `eq` and `is_sorted`
short-circuit on the first difference or inversion, so they too are safe on an infinite source.

## Outputs

`collect(container=tuple)` materialises into `tuple` (default), `list`, `set`, or `dict`:

```py
>>> from itrx import Itr
>>> Itr(("apple", "banana", "carrot")).groupby(len).collect(dict)
{5: ('apple',), 6: ('banana', 'carrot')}
```

- `collect(dict)` requires the iterator to yield 2-tuples — pair up with `zip`, `enumerate`,
  `groupby`, `chunk_by`, `value_counts` or a `map` that returns pairs first.
- `collect(set)` is the "unique" operation (and drops order).
- `for_each(func)` is the side-effecting terminal; `consume()` exhausts for side effects alone
  (pair it with `inspect`); `fold(init, func)` is the general terminal reduction.

## Method notes and gotchas

- **Single-pass, no rewinding.** Like any Python iterator, an `Itr` is consumed once. To use a
  sequence twice, use `copy()` (splits at the current position into an independent `Itr`),
  `tee(n)`, or re-wrap the original source. There is no `reset`.
- **`peek()` does not advance**, and returns the same value on repeated calls; it raises
  `StopIteration` when exhausted. `next_if(predicate)` consumes the next item **only** if it
  satisfies the predicate, otherwise leaves it in place and returns `None` (also `None` when
  exhausted) — the correct tool for conditional lookahead parsing.
- **`tee(n)` and `copy()` share a buffer.** After `tee`, do not consume the original `Itr` — use
  the returned ones. If one tee lags far behind another the buffer grows to hold the gap, so
  memory can blow up on a large or infinite source. `tee(n)` raises `ValueError` for `n < 1`.
- **`nth(n)` is 0-based** (like Rust's `Iterator::nth`): `nth(0)` is the first item. It raises
  `ValueError` for `n < 0` and `StopIteration` if the iterator is shorter than `n + 1`.
- **`next()` and `last()` raise `StopIteration`** on an empty iterator, they do not return `None`.
  `find(predicate)` *does* return `None` when nothing matches.
- **`groupby(key)` and `sorted_by(key)` sort the whole input up front**, so they reorder output,
  require mutually-orderable keys, and must not touch an infinite source. Use the lazy
  **`chunk_by(key)`** to group *consecutive* runs without sorting (the `itertools.groupby`
  semantics) — it preserves order and works on infinite iterators.
- **`value_counts()` is eager** and yields `(value, count)` most-common-first with ties by first
  appearance (pandas semantics, not sorted by key). Items must be hashable. Collect it into a
  `dict` for a lookup table.
- **`dedup()` removes only *adjacent* duplicates**, keeping the first of each run. It compares by
  equality (items need not be hashable) and stays lazy — it is not "unique". For global
  uniqueness use `collect(set)`, accepting the loss of order.
- **`dedup_with_count()` is run-length encoding** — the same adjacent-run logic as `dedup`, but
  yielding `(item, count)` pairs. It is the lazy, positional counterpart to `value_counts`: same
  output shape, but counting adjacent runs in source order rather than occurrences overall. On
  `[4, 4, 2, 3, 3, 1]` it gives `((4, 2), (2, 1), (3, 2), (1, 1))` where `value_counts()` gives
  `((4, 2), (3, 2), (2, 1), (1, 1))`. Prefer it to
  `chunk_by(f).map(lambda kv: (kv[0], len(kv[1])))`, which materialises every run to measure it.
  Note the `(item, count)` order is the reverse of Rust's `dedup_with_count`. It stays lazy on an
  infinite source, but an infinite individual run (e.g. `itertools.repeat(1)`) will hang.
- **`rev()` materialises the entire remaining sequence** into memory before yielding — unavoidable,
  but never call it on an unbounded source.
- **`repeat(n)` tees the iterator `n` times**, so it buffers the whole sequence for large `n`; it
  always returns a new `Itr` and leaves the original exhausted. `cycle()` repeats indefinitely.
- **`flatten()` removes exactly one level** of nesting and every item must itself be iterable
  (note that `str` is, which flattens strings into characters). `flat_map(f)` is
  `map(f).flatten()`.
- **`zip(other)` stops at the shorter input** (no `strict=True` mode); `zip_longest(other,
  fillvalue=...)` runs to the longer, padding. `interleave(other)` alternates and then yields the
  tail of whichever is longer.
- **`starmap(f)` unpacks each item as `*args`** — use it after `zip`, `enumerate`, `pairwise` or
  `product` instead of `map(lambda t: f(*t))`.
- **`map_dict(mapping)`** looks each item up in a `dict` (a `defaultdict` works); a missing key
  raises `KeyError`.
- **`batched(n)` yields non-overlapping tuples** (the last possibly short); **`rolling(n)`** yields
  overlapping windows of exactly `n` (a generalisation of `pairwise`); **`step_by(n)`** keeps every
  n-th item starting with the first.
- **`partition(predicate)`** returns `(matching, non_matching)` as two lazy `Itr`s built on a
  `copy()`, so the predicate runs twice per item and the same buffering caveat as `tee` applies.
- **`product(other)` materialises `other`** (as `itertools.product` does), so the *other* iterable
  must be finite even though the chain stays lazy in `self`.
- **`inspect(func)` is the lazy debugging hook** — it calls `func` on each item and passes it
  through unchanged, so you can drop it mid-chain without altering results.
- **`scan(init, func)` is `accumulate` with a separate state type and an early exit.** `func(state,
  item)` returns `(new_state, output)`, or `None` to stop. `None` as the *whole* return value halts;
  `(new_state, None)` yields `None` as an output, so the two are never ambiguous. Reach for
  `accumulate` when the state is just the running value, and `scan` otherwise.
- **`is_sorted(key=None, *, reverse=False)` is non-strict** — runs of equal items count as sorted,
  and empty/single-item iterators are sorted. The `key` argument covers Rust's `is_sorted_by_key`.
- **`eq(other)` compares contents; `==` does not.** `Itr` defines no `__eq__`, so `itr_a == itr_b`
  is an identity check. Use `.eq(other)` for an element-wise comparison, which also short-circuits
  rather than materialising both sides.

## Typing

`Itr[T]` is generic and every method is fully annotated, so type checkers track the element type
through a chain: `Itr(range(3)).map(str)` is `Itr[str]`, `.enumerate()` is `Itr[tuple[int, str]]`,
`.collect(list)` is `list[str]`. The package ships a `py.typed` marker. When a chain's inferred
type is wrong, the fix is usually to annotate the lambda or the source, not to cast the `Itr`.

## Working on itrx itself (not just using it)

If you are editing this library's own source (`src/itrx/itr.py`) rather than using it in a
downstream project, the repo's `AGENTS.md` governs the workflow. In short: no new runtime
dependencies (it is zero-dependency by design), new non-terminal methods must return an `Itr` and
stay lazy, every public method needs a doctest, the full gate suite is
`uv run ruff check && uv run ruff format --check && uv run ty check src && uv run pytest`, and
coverage is enforced at 100%. Regenerate `doc/apidoc.md` with
`uv run python src/scripts/introspect.py` whenever the public API changes.
