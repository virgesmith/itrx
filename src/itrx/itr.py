import itertools
import math
from collections import Counter, deque
from collections.abc import Callable, Generator, Iterable, Iterator
from typing import Any, TypeVar, cast, overload

T = TypeVar("T")
_CollectT = TypeVar("_CollectT")  # General item type for collected containers

Predicate = Callable[[T], bool]


class Itr[T](Iterator[T]):
    """A generic iterator adaptor class inspired by Rust's Iterator trait, providing a composable API for
    functional-style iteration and transformation over Python iterables.
    """

    def __init__(self, it: Iterable[T]) -> None:
        """Initialize the Itr with an iterable.

        Args:
            it (Iterable[T]): The iterable to wrap.

        """
        self._it = iter(it)

    def __iter__(self) -> Iterator[T]:
        "Implement the iter method of the Iterator protocol"
        return self

    def __next__(self) -> T:
        "Implement the next method of the Iterator protocol"
        return next(self._it)

    def __repr__(self) -> str:
        "Show the wrapped iterator, without consuming it"
        return f"Itr({self._it!r})"

    def accumulate(self, func: Callable[[T, T], T] | None = None, *, initial: T | None = None) -> "Itr[T]":
        """
        Return an iterator over the accumulated results of applying the function (or sum by default) to the items. Does
        not collapse the iterator like `reduce` or `fold`

        Args:
            func (Callable[[T, T], T] | None): A binary function to accumulate results. Defaults to addition.
            initial (T | None): An optional starting value. If specified, this value will be the first element of
            the resulting iterator

        Returns:
            Itr[T]: An iterator of accumulated results.

        Example:
            >>> list(Itr([1, 2, 3]).accumulate())
            [1, 3, 6]
            >>> list(Itr([2, 3, 4]).accumulate(lambda x, y: x * y))
            [2, 6, 24]
        """
        return Itr(itertools.accumulate(self._it, func, initial=initial))

    def all(self, predicate: Predicate[T] = bool) -> bool:
        """Return True if all elements in the iterator satisfy the predicate (by default, if all are truthy).

        Short-circuits at the first failure. An empty iterator returns True.

        Args:
            predicate (Callable[[T], bool]): A function to test each element. Defaults to `bool` (truthiness).

        Returns:
            bool: True if all elements satisfy the predicate, False otherwise.

        Example:
            >>> Itr([1, 2, 0]).all()
            False
            >>> Itr([2, 4, 6]).all(lambda x: x % 2 == 0)
            True
        """
        return all(map(predicate, self._it))

    def any(self, predicate: Predicate[T] = bool) -> bool:
        """Return True if any element in the iterator satisfies the predicate (by default, if any is truthy).

        Short-circuits at the first success. An empty iterator returns False.

        Args:
            predicate (Callable[[T], bool]): A function to test each element. Defaults to `bool` (truthiness).

        Returns:
            bool: True if any element satisfies the predicate, False otherwise.

        Example:
            >>> Itr([0, "", None]).any()
            False
            >>> Itr([1, 2, 3]).any(lambda x: x > 2)
            True
        """
        return any(map(predicate, self._it))

    def batched(self, n: int) -> "Itr[tuple[T, ...]]":
        """
        Groups the elements of the iterator into batches of size `n`.

        Args:
            n (int): The size of each batch. Must be at least 1.

        Returns:
            Itr[tuple[T, ...]]: An iterator yielding tuples of up to `n` elements from the original iterator.

        Raises:
            ValueError: If `n` is less than 1.

        Example:
            >>> list(Itr(range(7)).batched(3))
            [(0, 1, 2), (3, 4, 5), (6,)]
        """
        return cast("Itr[tuple[T, ...]]", Itr(itertools.batched(self._it, n)))

    def chain[U](self, *others: Iterable[U]) -> "Itr[T | U]":
        """Chain this iterator with one or more other iterables, yielding all items from self followed by all items from
        each of the others in turn.

        Args:
            *others (Iterable[U]): The iterables to chain on.

        Returns:
            Itr[T | U]: A new iterator yielding items from all the iterables.

        Example:
            >>> Itr([1, 2]).chain([3], (4, 5)).collect()
            (1, 2, 3, 4, 5)
        """
        return cast("Itr[T | U]", Itr(itertools.chain(self._it, *others)))

    @overload
    def collect(self, container: type[tuple[T, ...]] = tuple) -> tuple[T, ...]: ...
    @overload
    def collect(self, container: type[list[T]]) -> list[T]: ...
    @overload
    def collect(self, container: type[set[T]]) -> set[T]: ...
    @overload
    def collect[K, V](self, container: type[dict[K, V]]) -> dict[K, V]: ...

    def collect(self, container: type[_CollectT] = tuple) -> _CollectT:  # ty: ignore[invalid-parameter-default]
        """Collect all remaining items from the iterator into a container (tuple by default).

        Returns:
            _CollectT: The remaining items collected into the given container type.

        """
        return container(self._it)

    def compress(self, selectors: Iterable[object]) -> "Itr[T]":
        """Yield only the items whose corresponding selector is truthy (like `itertools.compress`).

        Stops when either the iterator or the selectors are exhausted.

        Args:
            selectors (Iterable[object]): Values whose truthiness decides whether the item in the same position is kept.

        Returns:
            Itr[T]: An iterator over the selected items.

        Example:
            >>> Itr("abcde").compress([1, 0, 1, 0, 1]).collect()
            ('a', 'c', 'e')
        """
        return Itr(itertools.compress(self._it, selectors))

    def consume(self) -> None:
        """Exhaust the iterator. Useful when only the side effects are required (see inspect)

        Do not use on an open-ended iterator

        Returns:
            None
        """
        deque(self._it, 0)

    def copy(self) -> "Itr[T]":
        """Splits the iterator at its *current state* into two independent iterators.

        Returns:
            Itr[T]: A new Itr instance wrapping one copy of the original iterator.

        """
        self._it, it = itertools.tee(self._it)
        return Itr(it)

    def count(self) -> int:
        """Count the number of remaining items in the iterator. NB Consumes the iterator.

        Returns:
            int: The number of remaining items.

        """
        return sum(1 for _ in self._it)

    def cycle(self) -> "Itr[T]":
        """
        Returns a new iterator that cycles indefinitely over the elements of the current iterator.

        Yields:
            Itr[T]: An iterator that repeats the elements of the original iterator endlessly.

        Example:
            >>> itr = Itr([1, 2, 3])
            >>> cycler = itr.cycle()
            >>> cycler.take(5).collect()
            (1, 2, 3, 1, 2)
        """
        return Itr(itertools.cycle(self._it))

    def dedup(self) -> "Itr[T]":
        """Lazily remove *consecutive* duplicate items, keeping the first of each run (like Rust's `dedup`).

        Only adjacent duplicates are removed, so the result may still contain repeated values if they are not
        contiguous. Items are compared by equality and do not need to be hashable. Works on infinite iterators.

        Returns:
            Itr[T]: An iterator over the items with consecutive duplicates removed.

        Example:
            >>> Itr([1, 1, 2, 2, 2, 3, 1]).dedup().collect()
            (1, 2, 3, 1)
        """
        return Itr(k for k, _ in itertools.groupby(self._it))

    def dedup_with_count(self) -> "Itr[tuple[T, int]]":
        """Lazily collapse each *consecutive* run of equal items into a (item, count) pair (run-length encoding).

        The lazy, positional counterpart to `value_counts`: this counts adjacent runs and preserves order (so the
        same item may appear more than once), where `value_counts` counts occurrences over the whole iterator and is
        eager. Items are compared by equality and do not need to be hashable. Works on infinite iterators, provided
        no individual run is infinite.

        Returns:
            Itr[tuple[T, int]]: An iterator of (item, run length) pairs.

        Example:
            >>> Itr([4, 4, 2, 3, 3, 1]).dedup_with_count().collect()
            ((4, 2), (2, 1), (3, 2), (1, 1))
        """
        # note the (item, count) ordering matches value_counts, and is the reverse of Rust's dedup_with_count
        return cast("Itr[tuple[T, int]]", Itr((k, sum(1 for _ in g)) for k, g in itertools.groupby(self._it)))

    def enumerate(self, *, start: int = 0) -> "Itr[tuple[int, T]]":
        """Yield pairs of (index, item) for each item in the iterator, where index starts at 0 or the value provided

        Args:
            start (int): The index of the first item. Defaults to 0.

        Returns:
            Itr[tuple[int, T]]: An iterator of (index, item) pairs.

        Example:
            >>> Itr("ab").enumerate(start=1).collect()
            ((1, 'a'), (2, 'b'))
        """
        return cast("Itr[tuple[int, T]]", Itr(enumerate(self._it, start)))

    def eq(self, other: Iterable[Any]) -> bool:
        """Compare the remaining items with another iterable, element by element (like Rust's `Iterator::eq`).

        Returns True only if both yield equal items in the same order and have the same length. Comparison
        short-circuits at the first difference, so unlike `tuple(a) == tuple(b)` neither side is fully materialised.
        NB This consumes as much of the iterator as it needs to. Note that `Itr` does not define `__eq__`, so `==`
        compares identity, not contents.

        Args:
            other (Iterable[Any]): The iterable to compare against.

        Returns:
            bool: True if the sequences are element-wise equal.

        Example:
            >>> Itr([1, 2, 3]).eq([1, 2, 3])
            True
            >>> Itr([1, 2, 3]).eq([1, 2])
            False
        """
        unequal = object()
        return all(a == b for a, b in itertools.zip_longest(self._it, other, fillvalue=unequal))

    def filter(self, predicate: Predicate[T]) -> "Itr[T]":
        """Yield only items that satisfy the predicate.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            Itr[T]: An iterator of filtered items.

        """
        return Itr(filter(predicate, self._it))

    def filter_map[U](self, mapper: Callable[[T], U | None]) -> "Itr[U]":
        """Map each item, keeping only the results that are not None (like Rust's `Iterator::filter_map`).

        Equivalent to `map(mapper).filter(lambda x: x is not None)`, but typed as `Itr[U]` rather than
        `Itr[U | None]`. Only None is dropped: other falsy results such as 0 or "" are kept.

        Args:
            mapper (Callable[[T], U | None]): A function mapping each item to a result, or None to drop it.

        Returns:
            Itr[U]: An iterator over the non-None results.

        Example:
            >>> Itr(["1", "x", "0"]).filter_map(lambda s: int(s) if s.isdigit() else None).collect()
            (1, 0)
        """
        return Itr(y for y in map(mapper, self._it) if y is not None)

    def find(self, predicate: Predicate[T]) -> T | None:
        """Return the first item in the iterator that satisfies the predicate, or None if not found.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            T | None: The first matching item, or None.

        """
        return next(filter(predicate, self._it), None)

    def find_map[U](self, mapper: Callable[[T], U | None]) -> U | None:
        """Return the first result of the mapper that is not None, or None if there is none (like Rust's
        `Iterator::find_map`).

        Equivalent to `filter_map(mapper).next()`, except that it returns None rather than raising when there is no
        result. Short-circuits at the first result.

        Args:
            mapper (Callable[[T], U | None]): A function mapping each item to a result, or None to skip it.

        Returns:
            U | None: The first non-None result, or None.

        Example:
            >>> Itr(["a", "12", "3"]).find_map(lambda s: int(s) if s.isdigit() else None)
            12
        """
        return next((y for y in map(mapper, self._it) if y is not None), None)

    def flat_map[U](self, mapper: Callable[[T], Iterable[U]]) -> "Itr[U]":
        """Map each item to an iterable, then flatten one level.

        Args:
            mapper (Callable[[T], Iterable[U]]): A function mapping each item to an iterable.

        Returns:
            Itr[U]: An iterator over the mapped and flattened items.

        """
        return Itr(itertools.chain.from_iterable(map(mapper, self._it)))

    def flatten[U](self) -> "Itr[U]":
        """Flatten one level of nesting in the iterator. Each item must itself be iterable.

        Returns:
            Itr[U]: An iterator over the flattened items.

        """
        return Itr(itertools.chain.from_iterable(cast("Iterable[Iterable[U]]", self._it)))

    def fold[U](self, init: U, func: Callable[[U, T], U]) -> U:
        """Reduce the iterator to a single value using a function and an initial value.

        Args:
            init (U): The initial value.
            func (Callable[[U, T], U]): The function to combine values.

        Returns:
            U: The final reduced value.

        """
        result = init
        for item in self._it:
            result = func(result, item)
        return result

    def for_each(self, func: Callable[[T], None]) -> None:
        """Apply a function to each item in the iterator.

        Args:
            func (Callable[[T], None]): The function to apply.

        """
        for item in self._it:
            func(item)

    def chunk_by[U](self, grouper: Callable[[T], U]) -> "Itr[tuple[U, tuple[T, ...]]]":
        """
        Group *consecutive* elements that share the same key, lazily. Unlike `groupby`, the input is not sorted, so
        only adjacent runs are grouped (mirroring `itertools.groupby` and Rust's `chunk_by`). This preserves order and
        works on infinite iterators.

        Args:
            grouper (Callable[[T], U]): The key function applied to each element.

        Returns:
            Itr[tuple[U, tuple[T, ...]]]: An iterator over (key, group) pairs, where each group is a tuple of the
            consecutive elements sharing that key.

        Example:
            >>> Itr([1, 1, 2, 3, 3, 1]).chunk_by(lambda x: x).map(lambda kv: kv[0]).collect()
            (1, 2, 3, 1)
        """
        key_fn = cast("Callable[[T], Any]", grouper)
        groups = ((k, tuple(v)) for k, v in itertools.groupby(self._it, key=key_fn))
        return cast("Itr[tuple[U, tuple[T, ...]]]", Itr(groups))

    def groupby[U](self, grouper: Callable[[T], U], *, sort: bool = True) -> "Itr[tuple[U, tuple[T,...]]]":
        """
        Sort and then group an iterable by the supplied key function. Note the following differences from itertools:
        - The iterable is pre-sorted because itertools.groupby only works correctly on sorted sequences
        - The resulting groupby objects are realised into tuples

        Because the input is sorted, this method is **eager**: it consumes and materialises the whole iterator
        immediately (so it must not be used on an infinite iterator), the output is ordered by key, and the keys must be
        mutually orderable. For lazy, order-preserving grouping of consecutive runs, see `chunk_by`.

        Semantically this is equivalent to pandas' `groupby`: all items sharing a key are collected into a single group
        regardless of their position, and `None` keys are not dropped. By default (`sort=True`) groups are emitted in
        sorted-key order, which requires mutually-orderable keys. With `sort=False` they are emitted in order of each
        key's first appearance, and the keys need only be hashable. Either way, the items within each group keep
        their original relative order.

        Args:
            grouper (Callable[[T], U]): The key function applied to each element.
            sort (bool): If True (the default), order the groups by key; otherwise by first appearance.

        Returns:
            Itr[tuple[U, tuple[T,...]]]: An iterator over the keys and tuples of values

        Example:
            >>> Itr(["bb", "a", "ccc", "dd"]).groupby(len).collect()
            ((1, ('a',)), (2, ('bb', 'dd')), (3, ('ccc',)))
            >>> Itr(["bb", "a", "ccc", "dd"]).groupby(len, sort=False).collect()
            ((2, ('bb', 'dd')), (1, ('a',)), (3, ('ccc',)))
        """
        if sort:
            key_fn = cast("Callable[[T], Any]", grouper)
            groups = ((k, tuple(v)) for k, v in itertools.groupby(sorted(self._it, key=key_fn), key=key_fn))
        else:
            groups_by_key: dict[U, list[T]] = {}
            for item in self._it:
                groups_by_key.setdefault(grouper(item), []).append(item)
            groups = ((k, tuple(v)) for k, v in groups_by_key.items())
        return cast("Itr[tuple[U, tuple[T, ...]]]", Itr(groups))

    def inspect(self, func: Callable[[T], None]) -> "Itr[T]":
        """
        Applies a function to each item in the iterator for side effects, yielding the original items unchanged.
        Useful for debugging

        Args:
            func (Callable[[T], None]): A function to apply to each item for side effects.

        Returns:
            Itr[T]: An iterator yielding the original items after applying the function.

        Example:
            >>> Itr([1, 2, 3]).inspect(print).consume()
            1
            2
            3
            >>>
        """

        def impl(x: T) -> T:
            func(x)
            return x

        return self.map(impl)

    def intersperse[U](self, item: U) -> "Itr[T | U]":
        """
        Yield items from the iterator, inserting the given item between each pair of items.

        Args:
            item (U): The item to intersperse.

        Returns:
            Itr[T | U]: An iterator with the item interspersed.

        """

        def intersperser(item: U) -> Generator[T | U, None, None]:
            try:
                current = next(self._it)
                while True:
                    yield current
                    current = next(self._it)
                    yield item
            except StopIteration:
                return

        return cast("Itr[T | U]", Itr(intersperser(item)))

    def interleave[U](self, *others: Iterable[U]) -> "Itr[T | U]":
        """
        Interleaves elements from this iterator with elements from one or more other iterables, taking one from each
        in turn (round-robin). Exhausted inputs are skipped, so the remaining elements of the longer ones are yielded
        in order.

        Args:
            *others (Iterable[U]): The iterables to interleave with.

        Returns:
            Itr[T | U]: A new iterator yielding elements from self and the others in turn.

        Example:
            >>> Itr([1, 3, 5]).interleave([2, 4, 6]).collect()
            (1, 2, 3, 4, 5, 6)
            >>> Itr([1, 3, 5, 7]).interleave([2, 4]).collect()
            (1, 2, 3, 4, 5, 7)
            >>> Itr([1, 4, 7]).interleave([2, 5], [3]).collect()
            (1, 2, 3, 4, 5, 7)
        """
        _sentinel = object()

        def interleaver() -> Generator[T | U, None, None]:
            for row in itertools.zip_longest(self._it, *others, fillvalue=_sentinel):
                for item in row:
                    if item is not _sentinel:
                        yield cast("T | U", item)

        return cast("Itr[T | U]", Itr(interleaver()))

    def is_sorted(self, key: Callable[[T], Any] | None = None, *, reverse: bool = False) -> bool:
        """Check whether the remaining items are in sorted order (like Rust's `Iterator::is_sorted`).

        Order is non-strict, so runs of equal items are sorted. An empty or single-item iterator is sorted. The
        check short-circuits at the first item out of order, but NB it consumes the iterator either way.

        Args:
            key (Callable[[T], Any] | None): Applied to each item before comparison, as in `sorted_by` (covering
                Rust's `is_sorted_by_key`). Defaults to comparing the items themselves.
            reverse (bool): If True, check for descending rather than ascending order.

        Returns:
            bool: True if the items are in the expected order.

        Example:
            >>> Itr([1, 2, 2, 3]).is_sorted()
            True
            >>> Itr(["ccc", "bb", "a"]).is_sorted(len, reverse=True)
            True
        """
        # T is unbounded so is not known to be orderable, as in sorted_by/groupby
        keyed = cast("Iterable[Any]", self._it if key is None else (key(item) for item in self._it))
        pairs = itertools.pairwise(keyed)
        return all(b <= a for a, b in pairs) if reverse else all(a <= b for a, b in pairs)

    def last(self) -> T:
        """Return the last item from the iterator. Do not use on an open-ended Iterable

        Returns:
            T: The last item.

        Raises:
            ValueError: If the iterator is empty.

        """
        *_, last_item = self._it
        return last_item

    def map[U](self, mapper: Callable[[T], U]) -> "Itr[U]":
        """Map each item in the iterator using the given function.

        Args:
            mapper (Callable[[T], U]): The function to apply.

        Returns:
            Itr[U]: An iterator of mapped items.

        """
        return Itr(map(mapper, self._it))

    def map_dict[U](self, mapper: dict[T, U]) -> "Itr[U]":
        """Map each item in the iterator using the given dictionary (supports defaultdict).

        Args:
            mapper (dict[T, U]): The lookup to apply.

        Returns:
            Itr[U]: An iterator of mapped items.

        """
        return Itr(mapper[m] for m in self._it)

    def map_while[U](self, predicate: Predicate[T], mapper: Callable[[T], U]) -> "Itr[U]":
        """Map each item in the iterator using the given function, while the predicate remains True.

        Args:
            predicate (Callable[[T], bool]): A function that takes an item and returns True to continue taking items, or False to stop.
            mapper (Callable[[T], U]): The function to apply.

        Returns:
            Itr[U]: An iterator of mapped items.

        """
        return Itr(map(mapper, itertools.takewhile(predicate, self._it)))

    def max(self, key: Callable[[T], Any] | None = None) -> T:
        """
        Return the maximum element from the iterator, optionally using a key function.

        Args:
            key (Callable[[T], Any] | None, optional): A function to extract a comparison key from each element. Defaults to None.

        Returns:
            T: The maximum element in the iterator.

        Raises:
            ValueError: If the iterator is empty.
        """
        return max(self._it, key=key)

    def min(self, key: Callable[[T], Any] | None = None) -> T:
        """
        Return the minimum element from the iterator, optionally using a key function.

        Args:
            key (Callable[[T], Any] | None, optional): A function to extract a comparison key from each element. Defaults to None.

        Returns:
            T: The minimum element in the iterator.

        Raises:
            ValueError: If the iterator is empty.
        """
        return min(self._it, key=key)

    def min_max(self, key: Callable[[T], Any] | None = None) -> tuple[T, T]:
        """Return the minimum and maximum elements in a single pass, optionally using a key function.

        Ties resolve as for the `min` and `max` builtins: the first minimal and the first maximal element are returned.
        NB Consumes the iterator.

        Args:
            key (Callable[[T], Any] | None, optional): A function to extract a comparison key from each element.
                Defaults to comparing the elements themselves.

        Returns:
            tuple[T, T]: The (minimum, maximum) elements.

        Raises:
            ValueError: If the iterator is empty.

        Example:
            >>> Itr([3, 1, 4, 1, 5]).min_max()
            (1, 5)
            >>> Itr(["bb", "a", "ccc", "dd"]).min_max(len)
            ('a', 'ccc')
        """
        try:
            lo = hi = next(self._it)
        except StopIteration:
            raise ValueError("min_max() of an empty iterator") from None
        # T is unbounded so is not known to be orderable, as in sorted_by/groupby
        key_fn = cast("Callable[[T], Any]", key if key is not None else lambda x: x)
        lo_key = hi_key = key_fn(lo)
        for item in self._it:
            k = key_fn(item)
            if k < lo_key:
                lo, lo_key = item, k
            elif k > hi_key:
                hi, hi_key = item, k
        return lo, hi

    def next(self) -> T:
        """Return the next item from the iterator, if available. Otherwise raises StopIteration

        Returns:
            T: The next item.

        """
        return next(self._it)

    def next_chunk(self, n: int) -> tuple[T, ...]:
        """Return a tuple of the next n items from the iterator.

        Args:
            n (int): The number of items to yield.

        Returns:
            tuple[T, ...]: The next n items (or fewer if the iterator is exhausted).

        """
        return self.take(n).collect()

    def next_if(self, predicate: Predicate[T]) -> T | None:
        """Consume and return the next item only if it satisfies the predicate (like Rust's `Peekable::next_if`).

        If the next item fails the predicate it is left in place (retrievable by a subsequent `next` or `peek`),
        and None is returned. None is also returned if the iterator is exhausted.

        Args:
            predicate (Callable[[T], bool]): A function to test the next item.

        Returns:
            T | None: The next item if it satisfies the predicate, otherwise None.

        Example:
            >>> it = Itr([1, 2, 3])
            >>> it.next_if(lambda x: x < 3)
            1
            >>> it.next_if(lambda x: x < 3)
            2
            >>> it.next_if(lambda x: x < 3) is None
            True
            >>> it.next()
            3
        """
        try:
            item = self.copy().next()
        except StopIteration:
            return None
        if predicate(item):
            return next(self._it)
        return None

    def nth(self, n: int) -> T:
        """Return the n-th item (0-based) from the iterator, consuming the preceding items.

        This matches Rust's ``Iterator::nth`` and Python's 0-based indexing conventions: ``nth(0)`` returns the first
        item, ``nth(1)`` the second, and so on.

        Args:
            n (int): The index (0-based) of the item to return.

        Returns:
            T: The n-th item.

        Raises:
            ValueError: if n < 0, or the iterator has fewer than n + 1 items.

        Example:
            >>> Itr("abc").nth(1)
            'b'
        """
        if n < 0:
            raise ValueError(f"nth index must be >= 0, got {n}")
        try:
            return next(itertools.islice(self._it, n, None))
        except StopIteration:
            raise ValueError(f"nth({n}) is out of range: the iterator has fewer than {n + 1} items") from None

    def pairwise(self) -> "Itr[tuple[T, T]]":
        """Returns an iterator that yields consecutive pairs of elements from the iterable.

        Each item produced is a tuple containing two consecutive elements from the original iterable.
        For example, given [1, 2, 3, 4], the `Itr` returned from this method yields (1, 2), (2, 3), (3, 4).

        Returns:
            Itr[tuple[T, T]]: An iterator over consecutive pairs from the original iterable.

        """
        return cast("Itr[tuple[T, T]]", Itr(itertools.pairwise(self._it)))

    def partition(self, predicate: Predicate[T]) -> tuple["Itr[T]", "Itr[T]"]:
        """
        Splits the elements of the iterator into two separate iterators based on a predicate.

        Args:
            predicate (Callable[[T], bool]): A function that takes an element and returns True or False.

        Returns:
            tuple[Itr[T], Itr[T]]: A tuple containing two iterators:
                - The first iterator yields elements for which the predicate returns True.
                - The second iterator yields elements for which the predicate returns False.

        Both iterators are lazy and share the source, and the predicate is called exactly once per item. Items destined
        for one iterator are buffered while the other is being consumed, so memory can grow if one side lags far behind
        (as with `tee`).

        Example:
            >>> even, odd = Itr(range(6)).partition(lambda x: x % 2 == 0)
            >>> even.collect(), odd.collect()
            ((0, 2, 4), (1, 3, 5))
        """
        matching: deque[T] = deque()
        non_matching: deque[T] = deque()

        def side(mine: deque[T]) -> Generator[T, None, None]:
            while True:
                while not mine:
                    try:
                        item = next(self._it)
                    except StopIteration:
                        return
                    (matching if predicate(item) else non_matching).append(item)
                yield mine.popleft()

        return Itr(side(matching)), Itr(side(non_matching))

    def peek(self) -> T:
        """Returns the next element in the sequence without advancing the iterator.

        Returns:
            T: The next element in the sequence.

        Raises:
            ValueError: If the iterator is exhausted.

        Note:
            This method copies the iterator to avoid modifying the original iterator's state. The copy is cheap
            even for repeated peeks: `itertools.tee` re-tees an already-teed iterator without adding a layer.

        Example:
            >>> it = Itr([1, 2])
            >>> it.peek(), it.peek()
            (1, 1)
            >>> it.next()
            1
            >>> it.peek()
            2
        """
        try:
            return self.copy().next()
        except StopIteration:
            raise ValueError("peek() on an exhausted iterator") from None

    def position(self, predicate: Predicate[T]) -> int | None:
        """
        Returns the index of the first element in the iterable that satisfies the given predicate, or None if there
        is no such element (like Rust's `Iterator::position`). The index counts from the iterator's current position.

        Args:
            predicate (Callable[[T], bool]): A function that takes an element and returns True if the element matches the condition.

        Returns:
            int | None: The index of the first matching element, or None.

        Example:
            >>> Itr("abc").position(lambda c: c == "b")
            1
            >>> Itr("abc").position(lambda c: c == "z") is None
            True
        """
        return next((i for i, item in enumerate(self._it) if predicate(item)), None)

    def prod(self, start: T | int = 1) -> T:
        """Return the product of `start` and all items in the iterator (so `start` if empty). NB Consumes the iterator.

        The items must support multiplication (e.g. numbers).

        Args:
            start (T | int): The initial value, multiplied by each item in turn. Defaults to 1.

        Returns:
            T: The product of all items.

        Example:
            >>> Itr([2, 3, 4]).prod()
            24
            >>> Itr([2, 3]).prod(10)
            60
        """
        return cast("T", math.prod(cast("Iterable[Any]", self._it), start=cast("Any", start)))

    def product[U](self, other: Iterable[U]) -> "Itr[tuple[T, U]]":
        """
        Creates a new iterator over the cartesian product of self and the other iterator

        Args:
            other (Iterable[U]): Another iterable.

        Returns:
            Itr[tuple[T, U]]: Iterator of 2-tuples with elements from each input iterator.
        """
        return cast("Itr[tuple[T, U]]", Itr(itertools.product(self._it, other)))

    def reduce(self, func: Callable[[T, T], T]) -> T:
        """Reduce the iterator to a single value using a function, taking the first item as the initial value.

        Args:
            func (Callable[[T, T], T]): The function to combine values.

        Returns:
            T: The final reduced value.

        Raises:
            ValueError: If the iterator is empty (use `fold` to supply an initial value instead).

        Example:
            >>> Itr([3, 1, 4]).reduce(max)
            4
        """
        try:
            first = next(self._it)
        except StopIteration:
            raise ValueError("reduce() of an empty iterator") from None
        return self.fold(first, func)

    def repeat(self, n: int) -> "Itr[T]":
        """
        Returns a new iterator that repeats the elements of the current iterator `n` times.

        Args:
            n (int): The number of times to repeat the elements.

        Returns:
            Itr[T]: An iterator yielding the elements of the original iterator repeated `n` times.

        Note:
            This implementation creates `n` independent iterators using `itertools.tee`, which may be inefficient for large `n` or large input iterators.
        """
        # this creates n iterators so may be inefficient
        return Itr(itertools.chain(*itertools.tee(self._it, n)))

    def rev(self) -> "Itr[T]":
        """Return a reversed iterator over the remaining items (materializes the sequence).

        Returns:
            Itr[T]: A reversed iterator.

        """
        # it's generally impossible to do this without materialising the entire sequence
        return Itr(tuple(self._it)[::-1])

    def rolling(self, n: int) -> "Itr[tuple[T, ...]]":
        """
        Rolling window (generalisation of pairwise)
        Rather than copying the iterator multiple times, collect n, yield the sequence and incrementally drop/add
        """
        if n < 1:
            raise ValueError(f"Invalid rolling window {n} (must be at least 1)")

        iterators = itertools.tee(self._it, n)
        shifted_iterators = (itertools.islice(it, i, None) for i, it in enumerate(iterators))
        return cast("Itr[tuple[T, ...]]", Itr(zip(*shifted_iterators, strict=False)))

    def scan[S, U](self, init: S, func: Callable[[S, T], tuple[S, U] | None]) -> "Itr[U]":
        """Lazily map items through a running state, optionally stopping early (like Rust's `Iterator::scan`).

        `func` receives the current state and the next item, and returns either a `(new_state, output)` pair or
        None to stop iterating. This generalises `accumulate`: the state need not be the same type as the items,
        and iteration can terminate on a condition. Yielding None as an *output* is unambiguous, since the halt
        signal is the entire return value rather than the output value.

        Args:
            init (S): The initial state.
            func (Callable[[S, T], tuple[S, U] | None]): Maps (state, item) to (new state, output), or None to stop.

        Returns:
            Itr[U]: An iterator over the outputs.

        Example:
            >>> Itr([1, 2, 3, 4]).scan(0, lambda total, x: (total + x, total + x)).collect()
            (1, 3, 6, 10)
            >>> Itr([1, 2, 3, 4]).scan(0, lambda total, x: None if total + x > 5 else (total + x, total + x)).collect()
            (1, 3)
        """

        def gen() -> Generator[U, None, None]:
            state = init
            for item in self._it:
                result = func(state, item)
                if result is None:
                    return
                state, output = result
                yield output

        return Itr(gen())

    def skip(self, n: int) -> "Itr[T]":
        """Skip the next n items in the iterator.

        Args:
            n (int): The number of items to skip.

        Returns:
            Itr[T]: An iterator over the remaining items.

        """
        return Itr(itertools.islice(self._it, n, None))

    def skip_while(self, predicate: Predicate[T]) -> "Itr[T]":
        """Skip items in the iterator as long as the predicate is true.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            Itr[T]: An iterator over the remaining items once the predicate first fails.

        """
        return Itr(itertools.dropwhile(predicate, self._it))

    def sorted_by(self, key: Callable[[T], Any] | None = None, *, reverse: bool = False) -> "Itr[T]":
        """Return an iterator over the items sorted by the given key function, or by the items themselves if no key is
        given.

        This method is **eager**: it consumes and materialises the whole iterator immediately (so it must not be
        used on an infinite iterator). The sort is stable: items that compare equal retain their relative order.

        Args:
            key (Callable[[T], Any] | None): A function to extract a comparison key from each item. Defaults to
                comparing the items themselves.
            reverse (bool): If True, sort in descending order. Defaults to False.

        Returns:
            Itr[T]: An iterator over the sorted items.

        Example:
            >>> Itr(["ccc", "a", "bb"]).sorted_by(len).collect()
            ('a', 'bb', 'ccc')
            >>> Itr([3, 1, 2]).sorted_by(reverse=True).collect()
            (3, 2, 1)
        """
        # T is unbounded so is not known to be orderable, as in groupby
        return Itr(sorted(cast("Iterable[Any]", self._it), key=key, reverse=reverse))

    def starmap[U](self, func: Callable[..., U]) -> "Itr[U]":
        """
        Applies a function to the elements of the iterator, unpacking the elements as arguments.

        Args:
            func (Callable[[T], U]): A function to apply to each element. Each element is expected to be an iterable of arguments for the function.

        Returns:
            Itr[U]: A new iterator with the results of applying the function to each unpacked element.

        Example:
            >>> itr = Itr([(1, 2), (3, 4)])
            >>> list(itr.starmap(lambda x, y: x + y))
            [3, 7]
        """
        return Itr(itertools.starmap(func, cast("Iterable[Iterable[Any]]", self._it)))

    def step_by(self, n: int) -> "Itr[T]":
        """Yield every n-th item from the iterator.

        Args:
            n (int): The step size.

        Returns:
            Itr[T]: An iterator yielding every n-th item.

        """
        return Itr(itertools.islice(self._it, 0, None, n))

    def sum(self, start: T | int = 0) -> T:
        """Return the sum of `start` and all items in the iterator (so `start` if empty). NB Consumes the iterator.

        The items must support addition with `start`: with the default of 0 that means numbers. Pass e.g. `start=[]` to
        concatenate lists (as with the `sum` builtin, strings are rejected: use `"".join` instead).

        Args:
            start (T | int): The initial value, to which each item is added in turn. Defaults to 0.

        Returns:
            T: The sum of all items.

        Example:
            >>> Itr([1, 2, 3]).sum()
            6
            >>> Itr([[1], [2, 3]]).sum([])
            [1, 2, 3]
        """
        return cast("T", sum(cast("Iterable[Any]", self._it), cast("Any", start)))

    def take(self, n: int) -> "Itr[T]":
        """Return an iterator over the next n items from the iterator.

        Args:
            n (int): The number of items to take.

        Returns:
            Itr[T]: An iterator over the next n items.

        """
        return Itr(itertools.islice(self._it, n))

    def take_last(self, n: int) -> "Itr[T]":
        """Return an iterator over the last n items (or all of them, if there are fewer than n).

        This method is **eager**: it consumes the whole iterator immediately to find its end (so it must not be used on
        an infinite iterator), but only ever holds n items in memory.

        Args:
            n (int): The number of items to keep.

        Returns:
            Itr[T]: An iterator over the last n items.

        Raises:
            ValueError: if n < 0

        Example:
            >>> Itr(range(10)).take_last(3).collect()
            (7, 8, 9)
        """
        if n < 0:
            raise ValueError(f"take_last requires n >= 0, got {n}")
        return Itr(deque(self._it, maxlen=n))

    def take_while(self, predicate: Predicate[T]) -> "Itr[T]":
        """Yield items from the iterator as long as the given predicate is true.

        Args:
            predicate (Callable[[T], bool]): A function that takes an item and returns True to continue taking items, or False to stop.

        Returns:
            Itr[T]: A new Itr yielding items while the predicate is true.

        """
        return Itr(itertools.takewhile(predicate, self._it))

    def tee(self, n: int = 2) -> tuple["Itr[T]", ...]:
        """
        Create multiple independent Itr wrappers that iterate over the same underlying iterator. NB Consuming any of the
        returned iterators will consume the original iterator

        This method calls itertools.tee on the wrapped iterator and returns a tuple of Itr
        objects, each wrapping one of the tee'd iterators. Each returned Itr yields the same
        sequence of items and can be consumed independently of the others.

        Args:
            n (int, optional): Number of independent iterators to create (default: 2). Must be >= 1.

        Returns:
            tuple[Itr[T], ...]: Tuple of length `n` containing the newly created Itr objects.

        Raises:
            ValueError: If `n` is less than 1.

        Notes:
        - The implementation uses itertools.tee; the tee'd iterators share internal buffers
          that store items produced by the original iterator until all tees have consumed them.
          If one or more returned iterators lag behind the others, buffered items will be
          retained and memory usage can grow.
        - After calling this method, avoid consuming the original wrapped iterator
          directly; use the returned Itr objects to prevent surprising interactions with the
          shared buffer.
        - Creating the tees is inexpensive, but the memory characteristics depend on how the
          resulting iterators are consumed relative to each other.

        Examples:
        >>> i = Itr(range(3))
        >>> a, b = i.tee(2)
        >>> list(a)
        [0, 1, 2]
        >>> list(b)
        [0, 1, 2]
        """
        if n < 1:
            raise ValueError(f"tee requires at least 1 iterator, got {n}")
        return tuple(Itr(t) for t in itertools.tee(self._it, n))

    def unique(self, key: Callable[[T], Any] | None = None) -> "Itr[T]":
        """Lazily drop items that have been seen before, keeping the first occurrence of each and preserving order.

        Unlike `dedup`, duplicates need not be adjacent; unlike `collect(set)`, order is preserved and it works on an
        infinite iterator. The seen items (or their keys) are held in a set, so they must be hashable, and memory
        grows with the number of distinct values.

        Args:
            key (Callable[[T], Any] | None): Applied to each item to decide uniqueness. Defaults to the item itself.

        Returns:
            Itr[T]: An iterator over the first occurrence of each distinct item (or key).

        Example:
            >>> Itr([3, 1, 3, 2, 1]).unique().collect()
            (3, 1, 2)
            >>> Itr(["apple", "avocado", "banana"]).unique(lambda s: s[0]).collect()
            ('apple', 'banana')
        """

        def gen() -> Generator[T, None, None]:
            seen: set[Any] = set()
            for item in self._it:
                k = item if key is None else key(item)
                if k not in seen:
                    seen.add(k)
                    yield item

        return Itr(gen())

    def unzip[U, V](self: "Itr[tuple[U, V]]") -> tuple["Itr[U]", "Itr[V]"]:
        """Splits the iterator of pairs into two separate iterators, each containing the elements from one position of
        the pairs.

        Returns:
            tuple[Itr[U], Itr[V]]: A tuple containing two Itr instances. The first contains all first elements,
            and the second contains all second elements from the original iterator of pairs.

        Note:
            This implementation does not materialize the entire iterator at once. It uses itertools.tee to split the iterator,
            and then maps over each to extract the respective elements.

        """
        it1, it2 = itertools.tee(self._it, 2)
        return Itr(x[0] for x in it1), Itr(x[1] for x in it2)

    def value_counts(self) -> "Itr[tuple[T, int]]":
        """
        Returns an iterator over the number of times distinct items appear in the original iterator, most common
        first (like pandas' `value_counts`). Ties are ordered by first appearance. Items must be hashable, and the
        result can be collected into a dict.

        This method is **eager**: it consumes the whole iterator immediately, so do not use it on an infinite
        iterator.

        Returns:
            Itr[tuple[T, int]]: An iterator of (value, count) pairs in descending count order.

        Example:
            >>> Itr("abracadabra").value_counts().collect()
            (('a', 5), ('b', 2), ('r', 2), ('c', 1), ('d', 1))
        """
        return cast("Itr[tuple[T, int]]", Itr(Counter(self._it).most_common()))

    def zip[U](self, other: Iterable[U], *, strict: bool = False) -> "Itr[tuple[T, U]]":
        """Yield pairs of items from this iterator and another iterable, stopping at the end of the shorter one.

        Args:
            other (Iterable[U]): The other iterable.
            strict (bool): If True, raise `ValueError` (when the shorter input runs out, since this is lazy) if the
                inputs differ in length, as with the `zip` builtin. Defaults to False.

        Returns:
            Itr[tuple[T, U]]: An iterator of paired items.

        Example:
            >>> Itr([1, 2, 3]).zip("ab").collect()
            ((1, 'a'), (2, 'b'))
        """
        return cast("Itr[tuple[T, U]]", Itr(zip(self._it, other, strict=strict)))

    def zip_longest[U, V](
        self, other: Iterable[U], *, fillvalue: V | None = None
    ) -> "Itr[tuple[T | V | None, U | V | None]]":
        """Yield pairs of items from this iterator and another iterable, padding the shorter with `fillvalue`.

        Unlike `zip`, iteration continues until the longer input is exhausted, with missing values replaced by
        `fillvalue`.

        Args:
            other (Iterable[U]): The other iterable.
            fillvalue (V | None): The value used to pad the shorter input. Defaults to None.

        Returns:
            Itr[tuple[T | V | None, U | V | None]]: An iterator of paired items.

        Example:
            >>> Itr([1, 2, 3]).zip_longest("ab", fillvalue="-").collect()
            ((1, 'a'), (2, 'b'), (3, '-'))
        """
        return cast(
            "Itr[tuple[T | V | None, U | V | None]]",
            Itr(itertools.zip_longest(self._it, other, fillvalue=fillvalue)),
        )
