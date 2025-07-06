from collections.abc import Generator
from typing import Never

import pytest

from itrx import Itr


def test_all_true() -> None:
    it = Itr([2, 4, 6])
    assert it.all(lambda x: x % 2 == 0)


def test_all_false() -> None:
    it = Itr([2, 3, 4])
    assert not it.all(lambda x: x % 2 == 0)


def test_any_true() -> None:
    it = Itr([1, 2, 3])
    assert it.any(lambda x: x == 2)


def test_any_false() -> None:
    it = Itr([1, 3, 5])
    assert not it.any(lambda x: x == 2)


def test_chain() -> None:
    it = Itr([1, 2]).chain([3, 4])
    assert it.collect() == (1, 2, 3, 4)


def test_collect() -> None:
    it = Itr([1, 2, 3])
    assert it.collect() == (1, 2, 3)


def test_count() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.count() == 4


def test_enumerate() -> None:
    it = Itr(["a", "b"]).enumerate()
    assert it.collect() == ((0, "a"), (1, "b"))


def test_filter() -> None:
    it = Itr([1, 2, 3, 4]).filter(lambda x: x % 2 == 0)
    assert it.collect() == (2, 4)


def test_find_found() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.find(lambda x: x > 2) == 3


def test_find_not_found() -> None:
    it = Itr([1, 2, 3])
    assert it.find(lambda x: x > 5) is None


def test_flatten() -> None:
    it = Itr([[1, 2], [3], [], [4, 5]])
    assert it.flatten().collect() == (1, 2, 3, 4, 5)


def test_flat_map() -> None:
    it: Itr[int] = Itr([[1, 2], [3]]).flat_map(lambda x: x * 10)  # type: ignore[operator]
    assert it.collect() == (10, 20, 30)


def test_fold() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.fold(0, lambda acc, x: acc + x) == 10


def test_for_each() -> None:
    result: list[int] = []
    it = Itr([1, 2, 3])
    it.for_each(result.append)
    assert result == [1, 2, 3]


def test_intersperse() -> None:
    it = Itr([1, 2, 3]).intersperse(0)
    assert it.collect() == (1, 0, 2, 0, 3)


def test_last() -> None:
    it = Itr([1, 2, 3])
    assert it.last() == 3


def test_map() -> None:
    it = Itr([1, 2, 3]).map(lambda x: x * 2)
    assert it.collect() == (2, 4, 6)


def test_next() -> None:
    it = Itr([10, 20, 30])
    assert it.next() == 10
    assert it.next() == 20


def test_next_rev() -> None:
    it = Itr([10, 20, 30]).rev()
    assert it.next() == 30
    assert it.next() == 20


def test_next_chunk() -> None:
    it = Itr([1, 2, 3, 4, 5])
    chunk = it.next_chunk(3)
    assert chunk == (1, 2, 3)


def test_next_chunk_overrun() -> None:
    it = Itr([1, 2, 3, 4, 5])
    assert it.next_chunk(10) == (1, 2, 3, 4, 5)


def test_nth() -> None:
    it = Itr([10, 20, 30, 40])
    assert it.nth(3) == 30
    assert it.nth(10) is None


def test_reduce() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.reduce(lambda a, b: a + b) == 10


def test_reduce_empty() -> None:
    it = Itr([1])
    # the lambda wont get called, result is just the single element
    assert it.reduce(lambda _a, _b: 0) == 1
    # and again on exhausted iterator
    with pytest.raises(StopIteration):
        it.reduce(lambda _a, _b: 0)


def test_skip() -> None:
    it = Itr([1, 2, 3, 4]).skip(2)
    assert it.collect() == (3, 4)


def test_step_by() -> None:
    it = Itr([1, 2, 3, 4, 5, 6]).step_by(2)
    assert it.collect() == (1, 3, 5)


def test_take() -> None:
    it = Itr([1, 2, 3, 4, 5])
    assert it.take(3).collect() == (1, 2, 3)


def test_zip() -> None:
    it = Itr([1, 2, 3]).zip(["a", "b", "c"])
    assert it.collect() == ((1, "a"), (2, "b"), (3, "c"))


def test_pairwise_basic() -> None:
    it = Itr([1, 2, 3, 4]).pairwise()
    assert it.collect() == ((1, 2), (2, 3), (3, 4))


def test_pairwise_empty() -> None:
    it = Itr[int]([]).pairwise()
    assert it.collect() == tuple()


def test_pairwise_single_element() -> None:
    it = Itr([42]).pairwise()
    assert it.collect() == tuple()


def test_pairwise_strings() -> None:
    it = Itr(["a", "b", "c"]).pairwise()
    assert it.collect() == (("a", "b"), ("b", "c"))


def test_pairwise_consumes_iterator() -> None:
    data = [10, 20, 30]
    it = Itr(data)
    pairs = it.pairwise().collect()
    assert pairs == ((10, 20), (20, 30))
    # After pairwise, original iterator is consumed
    assert it.collect() == ()


def test_copy() -> None:
    it = Itr(range(5))
    itc = it.copy()
    # consume the original
    it.collect()
    # copy is unaffected
    assert itc.next() == 0
    assert itc.next() == 1


def test_skip_while_some_skipped() -> None:
    it = Itr([1, 2, 3, 4, 5]).skip_while(lambda x: x < 3)
    assert it.collect() == (3, 4, 5)


def test_skip_while_none_skipped() -> None:
    it = Itr([1, 2, 3, 2, 1]).skip_while(lambda x: x < 3)
    assert it.collect() == (3, 2, 1)


def test_skip_while_all_skipped() -> None:
    it = Itr([1, 2, 3]).skip_while(lambda x: x < 10)
    # assert it.collect() == ()
    with pytest.raises(StopIteration):
        it.next()


def test_skip_while_empty_iterable() -> None:
    it = Itr[str]([]).skip_while(lambda x: True)
    assert it.collect() == ()


def test_skip_while_predicate_true_on_first() -> None:
    it = Itr([5, 6, 7]).skip_while(lambda x: x == 5)
    assert it.collect() == (6, 7)


def test_peek_does_not_advance_iterator() -> None:
    it = Itr([1, 2, 3])
    first_peek = it.peek()
    assert first_peek == 1
    # After peek, next should still return the same value
    assert it.next() == 1
    # Peek again, should return the next value
    assert it.peek() == 2
    assert it.next() == 2


def test_peek_on_empty_iterator_raises() -> None:
    it = Itr[bool]([])
    with pytest.raises(StopIteration):
        it.peek()


def test_take_while_some_true() -> None:
    it = Itr([1, 2, 3, 4, 1, 2])
    taken = it.take_while(lambda x: x < 4)
    assert taken.collect() == (1, 2, 3)


def test_take_while_all_true() -> None:
    it = Itr([1, 2, 3])
    taken = it.take_while(lambda x: x < 10)
    assert taken.collect() == (1, 2, 3)


def test_take_while_none_true() -> None:
    it = Itr([5, 6, 7])
    taken = it.take_while(lambda x: x < 0)
    assert taken.collect() == ()


def test_take_while_empty_iterable() -> None:
    it = Itr[float]([])
    taken = it.take_while(lambda x: True)
    assert taken.collect() == ()


def test_take_while_predicate_false_on_first() -> None:
    it = Itr([10, 20, 30])
    taken = it.take_while(lambda x: x < 10)
    assert taken.collect() == ()


def test_unzip_basic() -> None:
    it = Itr([(1, "a"), (2, "b"), (3, "c")])
    it1, it2 = it.unzip()  # type: ignore[var-annotated]
    assert it1.collect() == (1, 2, 3)
    assert it2.collect() == ("a", "b", "c")


def test_unzip_empty() -> None:
    it: Itr[int] = Itr([])
    with pytest.raises(ValueError):
        it.unzip()


def test_unzip_single_pair() -> None:
    it = Itr([(42, "x")])
    it1, it2 = it.unzip()  # type: ignore[var-annotated]
    assert it1.collect() == (42,)
    assert it2.collect() == ("x",)


def test_unzip_unequal_length_raises() -> None:
    class BadIter:
        def __iter__(self) -> Generator[tuple[int, ...], None, None]:
            yield (1, 2)
            yield (3,)  # Not a pair

    it = Itr(BadIter())
    with pytest.raises(ValueError):
        it.unzip()


def test_itr_iter_protocol() -> None:
    data = [1, 2, 3]
    it = Itr(data)
    # __iter__ should return an iterator
    iterator = iter(it)
    assert hasattr(iterator, "__next__")
    # __next__ should yield items in order
    assert next(it) == 1
    assert next(it) == 2
    assert next(it) == 3
    with pytest.raises(StopIteration):
        next(it)


def test_itr_iter_and_next_independent() -> None:
    data = [10, 20]
    it = Itr(data)
    # __iter__ returns the underlying iterator, so iter(it) is the same as it._it
    assert iter(it) is it._it
    # __next__ advances the iterator
    assert it.__next__() == 10
    assert it.__next__() == 20
    with pytest.raises(StopIteration):
        it.__next__()


def test_interleave_equal_length() -> None:
    it1 = Itr([1, 3, 5])
    it2 = Itr([2, 4, 6])
    result = it1.interleave(it2)
    assert result.collect() == (1, 2, 3, 4, 5, 6)


def test_interleave_first_longer() -> None:
    it1 = Itr([1, 3, 5, 7])
    it2 = Itr([2, 4])
    result = it1.interleave(it2)
    # Stops when either iterator is exhausted
    assert result.collect() == (1, 2, 3, 4)


def test_interleave_second_longer() -> None:
    it1 = Itr([1, 3])
    it2 = Itr([2, 4, 6, 8])
    result = it1.interleave(it2)
    assert result.collect() == (1, 2, 3, 4)


def test_interleave_empty_first() -> None:
    it1: Itr[int] = Itr([])
    it2 = Itr([2, 4, 6])
    result = it1.interleave(it2)
    assert result.collect() == ()


def test_interleave_empty_second() -> None:
    it1 = Itr([1, 3, 5])
    it2: Itr[int] = Itr([])
    result = it1.interleave(it2)
    assert result.collect() == ()


def test_interleave_both_empty() -> None:
    it1: Itr[int] = Itr([])
    it2: Itr[int] = Itr([])
    result = it1.interleave(it2)
    assert result.collect() == ()


def test_interleave_strings() -> None:
    it1 = Itr(["a", "c", "e"])
    it2 = Itr(["b", "d", "f"])
    result = it1.interleave(it2)
    assert result.collect() == ("a", "b", "c", "d", "e", "f")


def test_partition_basic() -> None:
    it = Itr([1, 2, 3, 4, 5, 6])
    even, odd = it.partition(lambda x: x % 2 == 0)
    assert even.collect() == (2, 4, 6)
    assert odd.collect() == (1, 3, 5)


def test_partition_all_true() -> None:
    it = Itr([2, 4, 6])
    even, odd = it.partition(lambda x: x % 2 == 0)
    assert even.collect() == (2, 4, 6)
    assert odd.collect() == ()


def test_partition_all_false() -> None:
    it = Itr([1, 3, 5])
    even, odd = it.partition(lambda x: x % 2 == 0)
    assert even.collect() == ()
    assert odd.collect() == (1, 3, 5)


def test_partition_empty() -> None:
    it: Itr[int] = Itr([])
    a, b = it.partition(lambda x: True)
    assert a.collect() == ()
    assert b.collect() == ()


def test_partition_predicate_side_effect() -> None:
    calls = []

    def pred(x: int) -> bool:
        calls.append(x)
        return x > 2

    it = Itr([1, 2, 3])
    a, b = it.partition(pred)
    # Both iterators should be independent and call the predicate as needed
    assert a.collect() == (3,)
    assert b.collect() == (1, 2)
    # The predicate should have been called for all elements twice (once per filter)
    assert calls == [1, 2, 3, 1, 2, 3]


def test_starmap_basic() -> None:
    it = Itr([(1, 2), (3, 4), (5, 6)]).starmap(lambda x, y: x + y)
    assert it.collect() == (3, 7, 11)


def test_starmap_with_mul() -> None:
    it = Itr([(2, 3), (4, 5)]).starmap(lambda x, y: x * y)
    assert it.collect() == (6, 20)


def test_starmap_empty() -> None:
    it = Itr([]).starmap(lambda *args: sum(args))
    assert it.collect() == ()


def test_starmap_single_tuple() -> None:
    it = Itr([(10, 20)]).starmap(lambda x, y: x - y)
    assert it.collect() == (-10,)


def test_starmap_raises_on_wrong_arity() -> None:
    it = Itr([(1, 2, 3)])
    with pytest.raises(TypeError):
        it.starmap(lambda x, y: x + y).collect()


def test_repeat_basic() -> None:
    it = Itr([1, 2, 3]).repeat(2)
    assert it.collect() == (1, 2, 3, 1, 2, 3)


def test_repeat_invalid() -> None:
    with pytest.raises(ValueError):
        Itr([1, 2, 3]).repeat(-1)


def test_repeat_zero() -> None:
    it = Itr([1, 2, 3]).repeat(0)
    # Repeating zero times should yield nothing
    assert it.collect() == ()


def test_repeat_one() -> None:
    it = Itr([4, 5]).repeat(1)
    # Repeating once should yield the original sequence
    assert it.collect() == (4, 5)


def test_repeat_empty_iterable() -> None:
    it: Itr[int] = Itr([]).repeat(3)
    # Repeating an empty iterable should yield nothing
    assert it.collect() == ()


def test_repeat_large_n() -> None:
    it = Itr([7]).repeat(5)
    assert it.collect() == (7, 7, 7, 7, 7)


def test_repeat_consumes_original() -> None:
    it = Itr([1, 2])
    repeated = it.repeat(2)
    # After repeat, the original iterator is exhausted
    assert repeated.collect() == (1, 2, 1, 2)
    assert it.collect() == ()


def test_itr_cycle_basic() -> None:
    it = Itr([1, 2, 3]).cycle()
    assert it.take(7).collect() == (1, 2, 3, 1, 2, 3, 1)


def test_itr_cycle_empty() -> None:
    it: Itr[int] = Itr([]).cycle()
    assert it.take(5).collect() == ()


def test_itr_rev_basic() -> None:
    it = Itr([1, 2, 3, 4]).rev()
    assert it.collect() == (4, 3, 2, 1)


def test_itr_rev_empty() -> None:
    it: Itr[int] = Itr([]).rev()
    assert it.collect() == ()


def test_itr_rev_single_element() -> None:
    it = Itr([42]).rev()
    assert it.collect() == (42,)


def test_max_basic() -> None:
    it = Itr([1, 5, 3, 2])
    assert it.max() == 5


def test_max_with_negative_numbers() -> None:
    it = Itr([-10, -5, -20])
    assert it.max() == -5


def test_max_single_element() -> None:
    it = Itr([42])
    assert it.max() == 42


def test_max_strings() -> None:
    it = Itr(["apple", "banana", "pear"])
    assert it.max() == "pear"


def test_max_empty_raises() -> None:
    it: Itr[int] = Itr([])
    with pytest.raises(ValueError):
        it.max()


def test_min_basic() -> None:
    it = Itr([1, 5, 3, 2])
    assert it.min() == 1


def test_min_with_negative_numbers() -> None:
    it = Itr([-10, -5, -20])
    assert it.min() == -20


def test_min_single_element() -> None:
    it = Itr([42])
    assert it.min() == 42


def test_min_strings() -> None:
    it = Itr(["apple", "banana", "pear"])
    assert it.min() == "apple"


def test_min_empty_raises() -> None:
    it: Itr[int] = Itr([])
    with pytest.raises(ValueError):
        it.min()


def test_batched() -> None:
    it = Itr(range(15)).batched(6)
    assert it.next() == tuple(range(6))
    assert it.next() == tuple(range(6, 12))
    assert it.next() == tuple(range(12, 15))
    with pytest.raises(StopIteration):
        it.next()


def test_batched_empty() -> None:
    it: Itr[tuple[Never, ...]] = Itr([]).batched(6)
    with pytest.raises(StopIteration):
        it.next()


def test_batched_invalid() -> None:
    with pytest.raises(ValueError):
        Itr([]).batched(0)


def test_groupby() -> None:
    it = Itr(range(20)).groupby(lambda n: n % 5)
    for i in range(5):
        assert it.next() == (i, tuple(range(i, 20, 5)))


def test_groupby_string() -> None:
    it = Itr(("apple", "banana", "carrot")).groupby(len)
    d: dict[int, tuple[str, ...]] = it.collect(dict)
    assert tuple(d.keys()) == (5, 6)
    assert d[5] == ("apple",)
    assert d[6] == ("banana", "carrot")


def test_rolling() -> None:
    it = Itr(range(5)).rolling(3)
    assert next(it) == (0, 1, 2)
    assert next(it) == (1, 2, 3)
    assert next(it) == (2, 3, 4)
    with pytest.raises(StopIteration):
        next(it)


def test_rolling_empty() -> None:
    with pytest.raises(ValueError):
        Itr(range(5)).rolling(0)

    it: Itr[tuple[int, ...]] = Itr(range(0)).rolling(3)
    with pytest.raises(StopIteration):
        next(it)

    it = Itr([0, 1]).rolling(3)
    with pytest.raises(StopIteration):
        next(it)

    it = Itr([0, 1, 2]).rolling(3)
    it.next()
    with pytest.raises(StopIteration):
        next(it)


def test_product() -> None:
    a = Itr("abc").product(range(2)).collect()
    assert len(a) == 6
    assert a[0] == ("a", 0)
    assert a[-1] == ("c", 1)


def test_product_empty() -> None:
    a = Itr("abc").product(range(0)).collect()
    assert len(a) == 0

    with pytest.raises(StopIteration):
        Itr("").product(range(0)).next()


def test_product_single_empty() -> None:
    a = Itr("a").product(range(0)).collect()
    assert len(a) == 0

    with pytest.raises(StopIteration):
        Itr("a").product(range(0)).next()
