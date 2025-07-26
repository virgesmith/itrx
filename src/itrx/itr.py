import itertools
from collections import deque
from collections.abc import Generator, Iterable
from itertools import pairwise
from typing import Callable, Iterator, Self, TypeVar, overload

T = TypeVar("T")
_CollectT = TypeVar("_CollectT")  # General item type for collected containers

Predicate = Callable[[T], bool]


class Itr[T](Iterable[T]):
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
        return self._it

    def __next__(self) -> T:
        "Implement the next method of the Iterator protocol"
        return next(self._it)

    def all(self, predicate: Predicate[T]) -> bool:
        """Return True if all elements in the iterator satisfy the predicate.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            bool: True if all elements satisfy the predicate, False otherwise.

        """
        return all(predicate(item) for item in self._it)

    def any(self, predicate: Predicate[T]) -> bool:
        """Return True if any element in the iterator satisfies the predicate.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            bool: True if any element satisfies the predicate, False otherwise.

        """
        return any(predicate(item) for item in self._it)

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
        if n < 1:
            raise ValueError("Batch size must be at least 1")
        return Itr(itertools.batched(self._it, n))

    def chain[U](self, other: Iterable[U]) -> "Itr[T | U]":
        """Chain this iterator with another iterable, yielding all items from self followed by all items from other.

        Args:
            other (Iterable[U]): Another iterable to chain.

        Returns:
            Itr[T | U]: A new iterator yielding items from both iterables.

        """
        return Itr(itertools.chain(self._it, other))

    @overload
    def collect(self, container: type[tuple[T, ...]] = tuple) -> tuple[T, ...]: ...
    @overload
    def collect(self, container: type[list[T]]) -> list[T]: ...
    @overload
    def collect(self, container: type[set[T]]) -> set[T]: ...
    @overload
    def collect[K, V](self, container: type[dict[K, V]]) -> dict[K, V]: ...

    def collect(self, container: type[_CollectT] = tuple) -> _CollectT:  # type: ignore[assignment]
        """Collect all remaining items from the iterator into a sequence (tuple by default).

        Returns:
            tuple[T]: A list of all remaining items.

        """
        return container(self._it)  # type: ignore[call-arg]

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

    def enumerate(self, *, start: int = 0) -> "Itr[tuple[int, T]]":
        """Yield pairs of (index, item) for each item in the iterator, where index starts at 0 or the value provided

        Returns:
            Itr[tuple[int, T]]: An iterator of (index, item) pairs.

        """
        return Itr(enumerate(self._it, start))

    def filter(self, predicate: Predicate[T]) -> "Itr[T]":
        """Yield only items that satisfy the predicate.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            Itr[T]: An iterator of filtered items.

        """
        return Itr(filter(predicate, self._it))

    def find(self, predicate: Predicate[T]) -> T | None:
        """Return the first item in the iterator that satisfies the predicate, or None if not found.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            T | None: The first matching item, or None.

        """
        try:
            while True:
                item = next(self._it)
                if predicate(item):
                    return item
        except StopIteration:
            return None

    # TODO fix the type annotations
    def flat_map[U, V](self, mapper: Callable[[U], V]) -> "Itr[V]":
        """Map each item to an iterable and flatten the results. Each item must itself be iterable.

        Args:
            mapper (Callable[[U], V]): A function mapping each item to an iterable.

        Returns:
            Itr[V]: An iterator over the mapped and flattened items.

        """

        def flat_mapper() -> Generator[V, None, None]:
            try:
                while True:
                    seq = next(self._it)
                    # TODO fix type annotations are remove this
                    assert isinstance(seq, Iterable)
                    iseq = iter(seq)
                    try:
                        while True:
                            yield mapper(next(iseq))
                    except StopIteration:
                        pass
            except StopIteration:
                return None

        return Itr(flat_mapper())

    def flatten[U](self) -> "Itr[U]":
        """Flatten one level of nesting in the iterator. Each item must itself be iterable.

        Returns:
            Itr[U]: An iterator over the flattened items.

        """
        return Itr(itertools.chain.from_iterable(self._it))  # type: ignore[arg-type]

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

    def groupby[U](self, grouper: Callable[[T], U]) -> "Itr[tuple[U, tuple[T,...]]]":
        """
        Sort and then group an iterable by the supplied key function. Note the following differences from itertools:
        - The iterable is pre-sorted because itertools.groupby only works correctly on sorted sequences
        - The resulting groupby objects are realised into tuples

        Returns:
            Itr[tuple[U, tuple[T,...]]]: An iterator over the keys and tuples of values

        """
        return Itr(itertools.groupby(sorted(self._it, key=grouper), key=grouper)).map(lambda g: (g[0], tuple(g[1])))  # type: ignore[arg-type]

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
                return None

        return Itr(intersperser(item))

    def interleave[U](self, other: Iterable[U]) -> "Itr[T | U]":
        """
        Interleaves elements from this iterator with elements from another iterator.
        Stops when either iterator is exhausted.

        Args:
            other (Itr[U]): Another iterator to interleave with.

        Returns:
            Itr[T | U]: A new iterator yielding elements alternately from self and other.

        Example:
            itr1 = Itr([1, 3, 5])
            itr2 = Itr([2, 4, 6])
            result = itr1.interleave(itr2)
            list(result)  # [1, 2, 3, 4, 5, 6]
        """

        return Itr(self.zip(other).flatten())

    def last(self) -> T | None:
        """Return the last item from the iterator. Do not use on an open-ended Iterable

        Returns:
            T: The last item.

        """
        last_item = None
        for item in self._it:
            last_item = item
        return last_item

    def map[U](self, mapper: Callable[[T], U]) -> "Itr[U]":
        """Map each item in the iterator using the given function.

        Args:
            mapper (Callable[[T], U]): The function to apply.

        Returns:
            Itr[U]: An iterator of mapped items.

        """
        return Itr(map(mapper, self._it))

    def max(self) -> T:
        """
        Returns the maximum element from the underlying iterable.

        Returns:
            T: The maximum element in the iterable.

        Raises:
            ValueError: If the iterable is empty.
        """
        # TODO T should have a "comparable" bound
        return max(self._it)  # type: ignore[type-var]

    def min(self) -> T:
        """
        Returns the minimum element from the underlying iterable.

        Returns:
            T_co: The smallest element in the iterable.

        Raises:
            ValueError: If the iterable is empty.
        """
        # TODO T should have a "comparable" bound
        return min(self._it)  # type: ignore[type-var]

    def next(self) -> T:
        """Return the next item from the iterator, if available. Otherwise raises StopIteration

        Returns:
            T: The next item.

        """
        return next(self._it)

    def next_chunk(self, n: int) -> tuple[T, ...]:
        """Return a list of the next n items from the iterator.

        Args:
            n (int): The number of items to yield.

        Returns:
            tuple[T, ...]: The next n items (or fewer if the iterator is exhausted).

        """
        return self.take(n).collect()

    def nth(self, n: int) -> T | None:
        """Return the n-th item (1-based) from the iterator, or None if out of range.

        Args:
            n (int): The index (1-based) of the item to return.

        Returns:
            T | None: The n-th item, or None.

        """
        try:
            for _ in range(n - 1):
                next(self._it)
            return next(self._it)
        except StopIteration:
            return None

    def pairwise(self) -> "Itr[tuple[T, T]]":
        """Returns an iterator that yields consecutive pairs of elements from the iterable.

        Each item produced is a tuple containing two consecutive elements from the original iterable.
        For example, given [1, 2, 3, 4], the `Itr` returned from this method yields (1, 2), (2, 3), (3, 4).

        Returns:
            Itr[tuple[T, T]]: An iterator over consecutive pairs from the original iterable.

        """
        return Itr(pairwise(self._it))

    def partition(self, predicate: Predicate[T]) -> tuple["Itr[T]", "Itr[T]"]:
        """
        Splits the elements of the iterator into two separate iterators based on a predicate.

        Args:
            predicate (Callable[[T], bool]): A function that takes an element and returns True or False.

        Returns:
            tuple[Itr[T], Itr[T]]: A tuple containing two iterators:
                - The first iterator yields elements for which the predicate returns True.
                - The second iterator yields elements for which the predicate returns False.
        """
        copy = self.copy()
        return self.filter(predicate), copy.filter(lambda i: not predicate(i))

    def peek(self) -> T:
        """Returns the next element in the sequence without advancing the iterator.

        Returns:
            T: The next element in the sequence.

        Note:
            This method creates a copy of the iterator to avoid modifying the original iterator's state.

        """
        return self.copy().next()

    def product[U](self, other: Iterable[U]) -> "Itr[tuple[T, U]]":
        """
        Creates a new iterator over tuples of the combinations of self and the other iterator
        """
        return Itr(itertools.product(self._it, other))

    def reduce(self, func: Callable[[T, T], T]) -> T:
        """Reduce the iterator to a single value using a function.
        Will raise StopIteration if the iterator is exhausted.

        Args:
            func (Callable[[T, T], T]): The function to combine values.

        Returns:
            T: The final reduced value.

        """
        result = next(self._it)
        for item in self._it:
            result = func(result, item)
        return result

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

        def roller(it: "Itr[T]") -> Generator[tuple[T, ...], None, None]:
            try:
                window = deque(it.take(n).collect(), maxlen=n)
                while True:
                    if len(window) == n:
                        yield tuple(window)
                    window.append(it.next())
            except StopIteration:
                return None

        return Itr(roller(self))

    def skip(self, n: int) -> Self:
        """Skip the next n items in the iterator.

        Args:
            n (int): The number of items to skip.

        Returns:
            Self: The iterator itself.

        """
        for _ in range(n):
            next(self._it)
        return self

    def skip_while(self, predicate: Predicate[T]) -> "Itr[T]":
        """Skip items in the iterator as long as the predicate is true, returning self.

        Args:
            predicate (Callable[[T], bool]): A function to test each element.

        Returns:
            Itr[T]: The iterator itself after skipping items.

        """
        return Itr(itertools.dropwhile(predicate, self._it))

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
        return Itr(itertools.starmap(func, self._it))  # type: ignore[arg-type]

    def step_by(self, n: int) -> "Itr[T]":
        """Yield every n-th item from the iterator.

        Args:
            n (int): The step size.

        Returns:
            Itr[T]: An iterator yielding every n-th item.

        """

        def stepper(n: int) -> Generator[T, None, None]:
            try:
                while True:
                    yield next(self._it)
                    for _ in range(n - 1):
                        next(self._it)
            except StopIteration:
                return None

        return Itr(stepper(n))

    def take(self, n: int) -> "Itr[T]":
        """Return an iterator over the next n items from the iterator.

        Args:
            n (int): The number of items to take.

        Returns:
            Itr[T]: An iterator over the next n items.

        """
        return Itr(itertools.islice(self._it, n))

    def take_while(self, predicate: Predicate[T]) -> "Itr[T]":
        """Collects and returns items from the iterator as long as the given predicate is true.

        Args:
            predicate (Callable[[T], bool]): A function that takes an item and returns True to continue taking items, or False to stop.

        Returns:
            Self: A new Itr instance containing the items taken while the predicate was true.

        """
        return Itr(itertools.takewhile(predicate, self._it))

    def unzip[U, V](self) -> tuple["Itr[U]", "Itr[V]"]:
        """Splits the iterator of pairs into two separate iterators, each containing the elements from one position of the pairs.

        Returns:
            tuple[Itr[T], Itr[U]]: A tuple containing two Itr instances. The first contains all first elements, and the second contains all second elements from the original iterator of pairs.

        Raises:
            ValueError: If the underlying iterator does not yield pairs of equal length (enforced by strict=True).

        """
        # TODO express that T is tuple[U, V]
        it1, it2 = zip(*self._it, strict=True)
        return Itr(it1), Itr(it2)

    def zip[U](self, other: Iterable[U]) -> "Itr[tuple[T, U]]":
        """Yield pairs of items from this iterator and another iterable.

        Args:
            other (Iterable[U]): The other iterable.

        Returns:
            Itr[tuple[T, U]]: An iterator of paired items.

        """
        return Itr(zip(self._it, other, strict=False))
