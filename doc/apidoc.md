# `Itr` v0.3.0 class documentation
A generic iterator adaptor class inspired by Rust's Iterator trait, providing a composable API for
functional-style iteration and transformation over Python iterables.
## Public methods

### `__init__`

Initialize the Itr with an iterable.

Args:
    it (Iterable[T]): The iterable to wrap.



### `__iter__`

Implement the iter method of the Iterator protocol

### `__next__`

Implement the next method of the Iterator protocol

### `accumulate`


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


### `all`

Return True if all elements in the iterator satisfy the predicate.

Args:
    predicate (Callable[[T], bool]): A function to test each element.

Returns:
    bool: True if all elements satisfy the predicate, False otherwise.



### `any`

Return True if any element in the iterator satisfies the predicate.

Args:
    predicate (Callable[[T], bool]): A function to test each element.

Returns:
    bool: True if any element satisfies the predicate, False otherwise.



### `batched`


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


### `chain`

Chain this iterator with another iterable, yielding all items from self followed by all items from other.

Args:
    other (Iterable[U]): Another iterable to chain.

Returns:
    Itr[T | U]: A new iterator yielding items from both iterables.



### `chunk_by`


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


### `collect`

Collect all remaining items from the iterator into a container (tuple by default).

Returns:
    _CollectT: The remaining items collected into the given container type.



### `consume`

Exhaust the iterator. Useful when only the side effects are required (see inspect)

Do not use on an open-ended iterator

Returns:
    None


### `copy`

Splits the iterator at its *current state* into two independent iterators.

Returns:
    Itr[T]: A new Itr instance wrapping one copy of the original iterator.



### `count`

Count the number of remaining items in the iterator. NB Consumes the iterator.

Returns:
    int: The number of remaining items.



### `cycle`


Returns a new iterator that cycles indefinitely over the elements of the current iterator.

Yields:
    Itr[T]: An iterator that repeats the elements of the original iterator endlessly.

Example:
    >>> itr = Itr([1, 2, 3])
    >>> cycler = itr.cycle()
    >>> cycler.take(5).collect()
    (1, 2, 3, 1, 2)


### `dedup`

Lazily remove *consecutive* duplicate items, keeping the first of each run (like Rust's `dedup`).

Only adjacent duplicates are removed, so the result may still contain repeated values if they are not
contiguous. Items are compared by equality and do not need to be hashable. Works on infinite iterators.

Returns:
    Itr[T]: An iterator over the items with consecutive duplicates removed.

Example:
    >>> Itr([1, 1, 2, 2, 2, 3, 1]).dedup().collect()
    (1, 2, 3, 1)


### `enumerate`

Yield pairs of (index, item) for each item in the iterator, where index starts at 0 or the value provided

Returns:
    Itr[tuple[int, T]]: An iterator of (index, item) pairs.



### `filter`

Yield only items that satisfy the predicate.

Args:
    predicate (Callable[[T], bool]): A function to test each element.

Returns:
    Itr[T]: An iterator of filtered items.



### `find`

Return the first item in the iterator that satisfies the predicate, or None if not found.

Args:
    predicate (Callable[[T], bool]): A function to test each element.

Returns:
    T | None: The first matching item, or None.



### `flat_map`

Map each item to an iterable, then flatten one level.

Args:
    mapper (Callable[[T], Iterable[U]]): A function mapping each item to an iterable.

Returns:
    Itr[U]: An iterator over the mapped and flattened items.



### `flatten`

Flatten one level of nesting in the iterator. Each item must itself be iterable.

Returns:
    Itr[U]: An iterator over the flattened items.



### `fold`

Reduce the iterator to a single value using a function and an initial value.

Args:
    init (U): The initial value.
    func (Callable[[U, T], U]): The function to combine values.

Returns:
    U: The final reduced value.



### `for_each`

Apply a function to each item in the iterator.

Args:
    func (Callable[[T], None]): The function to apply.



### `groupby`


Sort and then group an iterable by the supplied key function. Note the following differences from itertools:
- The iterable is pre-sorted because itertools.groupby only works correctly on sorted sequences
- The resulting groupby objects are realised into tuples

Because the input is sorted, this method is **eager**: it consumes and materialises the whole iterator
immediately (so it must not be used on an infinite iterator), the output is ordered by key, and the keys must be
mutually orderable. For lazy, order-preserving grouping of consecutive runs, see `chunk_by`.

Semantically this is equivalent to pandas' default `groupby` (i.e. `sort=True`): all items sharing a key are
collected into a single group regardless of their position, and groups are emitted in sorted-key order. It has
no `sort=False` (appearance-order) option, requires mutually-orderable keys, and does not drop `None` keys.

Returns:
    Itr[tuple[U, tuple[T,...]]]: An iterator over the keys and tuples of values



### `inspect`


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


### `interleave`


Interleaves elements from this iterator with elements from another iterable, yielding alternately from each.
When one iterable is exhausted, the remaining elements of the other are yielded in order.

Args:
    other (Iterable[U]): Another iterable to interleave with.

Returns:
    Itr[T | U]: A new iterator yielding elements alternately from self and other.

Example:
    >>> Itr([1, 3, 5]).interleave([2, 4, 6]).collect()
    (1, 2, 3, 4, 5, 6)
    >>> Itr([1, 3, 5, 7]).interleave([2, 4]).collect()
    (1, 2, 3, 4, 5, 7)


### `intersperse`


Yield items from the iterator, inserting the given item between each pair of items.

Args:
    item (U): The item to intersperse.

Returns:
    Itr[T | U]: An iterator with the item interspersed.



### `last`

Return the last item from the iterator. Do not use on an open-ended Iterable

Returns:
    T: The last item.

Raises:
    ValueError: If the iterator is empty.



### `map`

Map each item in the iterator using the given function.

Args:
    mapper (Callable[[T], U]): The function to apply.

Returns:
    Itr[U]: An iterator of mapped items.



### `map_dict`

Map each item in the iterator using the given dictionary (supports defaultdict).

Args:
    mapper (dict[T, U]): The lookup to apply.

Returns:
    Itr[U]: An iterator of mapped items.



### `map_while`

Map each item in the iterator using the given function, while the predicate remains True.

Args:
    predicate (Callable[[T], bool]): A function that takes an item and returns True to continue taking items, or False to stop.
    mapper (Callable[[T], U]): The function to apply.

Returns:
    Itr[U]: An iterator of mapped items.



### `max`


Return the maximum element from the iterator, optionally using a key function.

Args:
    key (Callable[[T], Any] | None, optional): A function to extract a comparison key from each element. Defaults to None.

Returns:
    T: The maximum element in the iterator.

Raises:
    ValueError: If the iterator is empty.


### `min`


Return the minimum element from the iterator, optionally using a key function.

Args:
    key (Callable[[T], Any] | None, optional): A function to extract a comparison key from each element. Defaults to None.

Returns:
    T: The minimum element in the iterator.

Raises:
    ValueError: If the iterator is empty.


### `next`

Return the next item from the iterator, if available. Otherwise raises StopIteration

Returns:
    T: The next item.



### `next_chunk`

Return a tuple of the next n items from the iterator.

Args:
    n (int): The number of items to yield.

Returns:
    tuple[T, ...]: The next n items (or fewer if the iterator is exhausted).



### `next_if`

Consume and return the next item only if it satisfies the predicate (like Rust's `Peekable::next_if`).

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


### `nth`

Return the n-th item (0-based) from the iterator, consuming the preceding items.

This matches Rust's ``Iterator::nth`` and Python's 0-based indexing conventions: ``nth(0)`` returns the first
item, ``nth(1)`` the second, and so on.

Args:
    n (int): The index (0-based) of the item to return.

Returns:
    T: The n-th item.

Raises:
    StopIteration: if the iterator has fewer than n + 1 items.
    ValueError: if n < 0



### `pairwise`

Returns an iterator that yields consecutive pairs of elements from the iterable.

Each item produced is a tuple containing two consecutive elements from the original iterable.
For example, given [1, 2, 3, 4], the `Itr` returned from this method yields (1, 2), (2, 3), (3, 4).

Returns:
    Itr[tuple[T, T]]: An iterator over consecutive pairs from the original iterable.



### `partition`


Splits the elements of the iterator into two separate iterators based on a predicate.

Args:
    predicate (Callable[[T], bool]): A function that takes an element and returns True or False.

Returns:
    tuple[Itr[T], Itr[T]]: A tuple containing two iterators:
        - The first iterator yields elements for which the predicate returns True.
        - The second iterator yields elements for which the predicate returns False.


### `peek`

Returns the next element in the sequence without advancing the iterator.

The element is held in a single-item lookahead buffer (like Rust's `Peekable`), so repeated calls are O(1)
and return the same item until the iterator is advanced.

Returns:
    T: The next element in the sequence.

Raises:
    StopIteration: If the iterator is exhausted.

Example:
    >>> it = Itr([1, 2])
    >>> it.peek(), it.peek()
    (1, 1)
    >>> it.next()
    1
    >>> it.peek()
    2


### `position`


Returns the index of the first element in the iterable that satisfies the given predicate.

Args:
    predicate (Callable[[T], bool]): A function that takes an element and returns True if the element matches the condition.

Returns:
    int: The index of the first matching element.

Raises:
    StopIteration: If no element satisfies the predicate.


### `prod`

Return the product of all items in the iterator (1 if empty). NB Consumes the iterator.

The items must support multiplication (e.g. numbers).

Returns:
    T: The product of all items.

Example:
    >>> Itr([2, 3, 4]).prod()
    24


### `product`


Creates a new iterator over the cartesian product of self and the other iterator

Args:
    other (Iterable[U]): Another iterable.

Returns:
    Itr[tuple[T, U]]: Iterator of 2-tuples with elements from each input iterator.


### `reduce`

Reduce the iterator to a single value using a function.
Will raise StopIteration if the iterator is exhausted.

Args:
    func (Callable[[T, T], T]): The function to combine values.

Returns:
    T: The final reduced value.



### `repeat`


Returns a new iterator that repeats the elements of the current iterator `n` times.

Args:
    n (int): The number of times to repeat the elements.

Returns:
    Itr[T]: An iterator yielding the elements of the original iterator repeated `n` times.

Note:
    This implementation creates `n` independent iterators using `itertools.tee`, which may be inefficient for large `n` or large input iterators.


### `rev`

Return a reversed iterator over the remaining items (materializes the sequence).

Returns:
    Itr[T]: A reversed iterator.



### `rolling`


Rolling window (generalisation of pairwise)
Rather than copying the iterator multiple times, collect n, yield the sequence and incrementally drop/add


### `skip`

Skip the next n items in the iterator.

Args:
    n (int): The number of items to skip.

Returns:
    Itr[T]: An iterator over the remaining items.



### `skip_while`

Skip items in the iterator as long as the predicate is true.

Args:
    predicate (Callable[[T], bool]): A function to test each element.

Returns:
    Itr[T]: An iterator over the remaining items once the predicate first fails.



### `sorted_by`

Return an iterator over the items sorted by the given key function.

This method is **eager**: it consumes and materialises the whole iterator immediately (so it must not be
used on an infinite iterator). The sort is stable: items that compare equal retain their relative order.

Args:
    key (Callable[[T], Any]): A function to extract a comparison key from each item.
    reverse (bool): If True, sort in descending order. Defaults to False.

Returns:
    Itr[T]: An iterator over the sorted items.

Example:
    >>> Itr(["ccc", "a", "bb"]).sorted_by(len).collect()
    ('a', 'bb', 'ccc')


### `starmap`


Applies a function to the elements of the iterator, unpacking the elements as arguments.

Args:
    func (Callable[[T], U]): A function to apply to each element. Each element is expected to be an iterable of arguments for the function.

Returns:
    Itr[U]: A new iterator with the results of applying the function to each unpacked element.

Example:
    >>> itr = Itr([(1, 2), (3, 4)])
    >>> list(itr.starmap(lambda x, y: x + y))
    [3, 7]


### `step_by`

Yield every n-th item from the iterator.

Args:
    n (int): The step size.

Returns:
    Itr[T]: An iterator yielding every n-th item.



### `sum`

Return the sum of all items in the iterator (0 if empty). NB Consumes the iterator.

The items must support addition with int (e.g. numbers; use `reduce` or `fold` for other types).

Returns:
    T: The sum of all items.

Example:
    >>> Itr([1, 2, 3]).sum()
    6


### `take`

Return an iterator over the next n items from the iterator.

Args:
    n (int): The number of items to take.

Returns:
    Itr[T]: An iterator over the next n items.



### `take_while`

Yield items from the iterator as long as the given predicate is true.

Args:
    predicate (Callable[[T], bool]): A function that takes an item and returns True to continue taking items, or False to stop.

Returns:
    Itr[T]: A new Itr yielding items while the predicate is true.



### `tee`


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


### `unzip`

Splits the iterator of pairs into two separate iterators, each containing the elements from one position of
the pairs.

Returns:
    tuple[Itr[U], Itr[V]]: A tuple containing two Itr instances. The first contains all first elements,
    and the second contains all second elements from the original iterator of pairs.

Note:
    This implementation does not materialize the entire iterator at once. It uses itertools.tee to split the iterator,
    and then maps over each to extract the respective elements.



### `value_counts`


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


### `zip`

Yield pairs of items from this iterator and another iterable.

Args:
    other (Iterable[U]): The other iterable.

Returns:
    Itr[tuple[T, U]]: An iterator of paired items.



### `zip_longest`

Yield pairs of items from this iterator and another iterable, padding the shorter with `fillvalue`.

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

