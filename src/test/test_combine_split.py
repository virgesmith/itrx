from typing import Never

import pytest

from itrx import Itr


def test_chain() -> None:
    it = Itr([1, 2]).chain([3, 4])
    assert it.collect() == (1, 2, 3, 4)


def test_intersperse() -> None:
    it = Itr([1, 2, 3]).intersperse(0)
    assert it.collect() == (1, 0, 2, 0, 3)


def test_pairwise_basic() -> None:
    it = Itr([1, 2, 3, 4]).pairwise()
    assert it.collect() == ((1, 2), (2, 3), (3, 4))


def test_pairwise_empty() -> None:
    it = Itr[int]([]).pairwise()
    assert it.collect() == ()


def test_pairwise_single_element() -> None:
    it = Itr([42]).pairwise()
    assert it.collect() == ()


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
    a, b = it.partition(lambda _: True)
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

    with pytest.raises(ValueError):
        Itr(range(5)).batched(-1).collect()


def test_batched_invalid() -> None:
    with pytest.raises(ValueError):
        Itr([]).batched(0)


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


def test_inspect() -> None:
    total = 0

    def log(x: int) -> None:
        nonlocal total
        total += x

    a = Itr(range(10)).inspect(log).exhaust()
    assert a is None
    assert total == 45


def test_unzip_basic() -> None:
    data = [(1, "a"), (2, "b"), (3, "c")]
    itr = Itr(data)
    first, second = itr.unzip()  # type: ignore[var-annotated]
    assert tuple(first) == (1, 2, 3)
    assert tuple(second) == ("a", "b", "c")


def test_unzip_empty() -> None:
    itr = Itr([])  # type: ignore[var-annotated]
    first, second = itr.unzip()  # type: ignore[var-annotated]
    assert tuple(first) == ()
    assert tuple(second) == ()


def test_unzip_iterator_consumption() -> None:
    data = [(10, 20), (30, 40)]
    itr = Itr(data)
    first, second = itr.unzip()  # type: ignore[var-annotated]
    # Both can be iterated independently
    assert next(first) == 10
    assert next(second) == 20
    assert next(first) == 30
    assert next(second) == 40
    with pytest.raises(StopIteration):
        next(first)
    with pytest.raises(StopIteration):
        next(second)


def test_unzip_with_different_types() -> None:
    data = [(1, None), (2, 3.5), (3, "x")]
    itr = Itr(data)
    first, second = itr.unzip()  # type: ignore[var-annotated]
    assert tuple(first) == (1, 2, 3)
    assert tuple(second) == (None, 3.5, "x")


def test_zip_basic() -> None:
    a = Itr([1, 2, 3])
    b = [4, 5, 6]
    result = a.zip(b).collect(list)
    assert result == [(1, 4), (2, 5), (3, 6)]


def test_zip_left_shorter() -> None:
    a = Itr([1, 2])
    b = [3, 4, 5]
    result = a.zip(b).collect(list)
    assert result == [(1, 3), (2, 4)]


def test_zip_right_shorter() -> None:
    a = Itr([1, 2, 3])
    b = [4]
    result = a.zip(b).collect(list)
    assert result == [(1, 4)]


def test_zip_empty_left() -> None:
    a = Itr([])  # type: ignore[var-annotated]
    b = [1, 2, 3]
    result = a.zip(b).collect(list)
    assert result == []


def test_zip_empty_right() -> None:
    a = Itr([1, 2, 3])
    b = []  # type: ignore[var-annotated]
    result = a.zip(b).collect(list)
    assert result == []


def test_zip_both_empty() -> None:
    a = Itr([])  # type: ignore[var-annotated]
    b = []  # type: ignore[var-annotated]
    result = a.zip(b).collect(list)
    assert result == []


def test_zip_with_different_types() -> None:
    a = Itr([1, 2, 3])
    b = ["a", "b", "c"]
    result = a.zip(b).collect(list)
    assert result == [(1, "a"), (2, "b"), (3, "c")]
