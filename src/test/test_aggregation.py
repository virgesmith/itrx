import itertools
from operator import add, mul, sub, truediv

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


def test_count() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.count() == 4


def test_find_found() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.find(lambda x: x > 2) == 3


def test_find_not_found() -> None:
    it = Itr([1, 2, 3])
    assert it.find(lambda x: x > 5) is None


def test_fold() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.fold(0, lambda acc, x: acc + x) == 10


def test_last() -> None:
    it = Itr([1, 2, 3])
    assert it.last() == 3


def test_last_exhausted() -> None:
    it = Itr([1, 2, 3])
    it.consume()
    with pytest.raises(ValueError):
        it.last()


def test_reduce() -> None:
    it = Itr([1, 2, 3, 4])
    assert it.copy().reduce(lambda a, b: a + b) == 10

    assert it.copy().reduce(add) == 10
    assert it.copy().reduce(sub) == -8
    assert it.copy().reduce(mul) == 24
    assert it.copy().reduce(truediv) == 1 / 24


def test_reduce_empty() -> None:
    it = Itr([1])
    # the lambda wont get called, result is just the single element
    assert it.reduce(lambda _a, _b: 0) == 1
    # and again on exhausted iterator
    with pytest.raises(TypeError, match="empty"):
        it.reduce(lambda _a, _b: 0)


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


def test_min_max_key() -> None:
    d = {"a": 1, "b": 2, "c": 3, "d": 0}

    assert Itr(d).max(key=d.get) == "c"
    assert Itr(d).min(key=d.get) == "d"


def test_value_counts_basic() -> None:
    data = [1, 2, 2, 3, 1, 4, 2]
    result = Itr(data).value_counts().collect(dict)
    assert result == {1: 2, 2: 3, 3: 1, 4: 1}


def test_value_counts_most_common_first() -> None:
    # like pandas: descending count, ties in first-appearance order
    data = [1, 2, 2, 3, 1, 4, 2]
    result = Itr(data).value_counts().collect()
    assert result == ((2, 3), (1, 2), (3, 1), (4, 1))


def test_value_counts_unorderable_items() -> None:
    # items only need to be hashable, not orderable
    data = [1j, 2j, 1j]
    result = Itr(data).value_counts().collect()
    assert result == ((1j, 2), (2j, 1))


def test_value_counts_empty() -> None:
    data: list[int] = []
    result = Itr(data).value_counts().collect(dict)
    assert result == {}


def test_value_counts_all_unique() -> None:
    data = [1, 2, 3, 4, 5]
    result = Itr(data).value_counts().collect(dict)
    assert result == {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}


def test_value_counts_all_duplicates() -> None:
    data = [7, 7, 7, 7]
    result = Itr(data).value_counts().collect(dict)
    assert result == {7: 4}


def test_value_counts_strings() -> None:
    data = ["a", "b", "a", "c", "b", "b"]
    result = Itr(data).value_counts().collect(dict)
    assert result == {"a": 2, "b": 3, "c": 1}


def test_value_counts_consumes_iterator() -> None:
    data = [1, 2, 2, 3]
    itr = Itr(data)
    _ = itr.value_counts()
    with pytest.raises(StopIteration):
        itr.next()


def test_sum() -> None:
    assert Itr([1, 2, 3]).sum() == 6
    assert Itr([1.5, 2.5]).sum() == 4.0


def test_sum_empty() -> None:
    assert Itr[int]([]).sum() == 0


def test_sum_consumes_iterator() -> None:
    it = Itr([1, 2, 3])
    assert it.sum() == 6
    with pytest.raises(StopIteration):
        it.next()


def test_prod() -> None:
    assert Itr([2, 3, 4]).prod() == 24
    assert Itr([0.5, 4.0]).prod() == 2.0


def test_prod_empty() -> None:
    assert Itr[int]([]).prod() == 1


def test_prod_with_zero() -> None:
    assert Itr([1, 0, 5]).prod() == 0


def test_eq() -> None:
    assert Itr([1, 2, 3]).eq([1, 2, 3])
    assert not Itr([1, 2, 3]).eq([1, 2, 4])


def test_eq_empty() -> None:
    assert Itr[int]([]).eq([])
    assert not Itr[int]([]).eq([1])


def test_eq_differing_lengths() -> None:
    assert not Itr([1, 2, 3]).eq([1, 2])
    assert not Itr([1, 2]).eq([1, 2, 3])


def test_eq_accepts_any_iterable() -> None:
    assert Itr("abc").eq(iter("abc"))
    assert Itr([1, 2, 3]).eq(Itr(range(1, 4)))


def test_eq_short_circuits_on_infinite() -> None:
    # differs at the third item, so neither side is exhausted
    assert not Itr(itertools.count()).eq(itertools.chain([0, 1], [99]))


def test_is_sorted() -> None:
    assert Itr([1, 2, 3]).is_sorted()
    assert not Itr([1, 3, 2]).is_sorted()


def test_is_sorted_allows_equal_items() -> None:
    assert Itr([1, 2, 2, 3]).is_sorted()


def test_is_sorted_empty_and_single() -> None:
    assert Itr[int]([]).is_sorted()
    assert Itr([5]).is_sorted()


def test_is_sorted_with_key() -> None:
    assert Itr(["a", "bb", "ccc"]).is_sorted(len)
    assert not Itr(["ccc", "a"]).is_sorted(len)


def test_is_sorted_reverse() -> None:
    assert Itr([3, 2, 1]).is_sorted(reverse=True)
    assert not Itr([1, 2, 3]).is_sorted(reverse=True)
    assert Itr(["ccc", "bb", "a"]).is_sorted(len, reverse=True)


def test_is_sorted_agrees_with_sorted_by() -> None:
    data = ["ccc", "a", "bb"]
    assert Itr(data).sorted_by(len).is_sorted(len)


def test_all_any_default_truthiness() -> None:
    assert Itr([1, "a", [0]]).all()
    assert not Itr([1, 0, 2]).all()
    assert Itr[int]([]).all()
    assert Itr([0, "", None, 3]).any()
    assert not Itr([0, "", None]).any()
    assert not Itr[int]([]).any()


def test_all_any_short_circuit() -> None:
    it = Itr(itertools.count())
    assert not it.all()  # 0 is falsy
    assert it.next() == 1
    assert Itr(itertools.count()).any()


def test_sum_prod_start() -> None:
    assert Itr([1, 2, 3]).sum(10) == 16
    assert Itr[int]([]).sum(10) == 10
    assert Itr([[1], [2, 3]]).sum([]) == [1, 2, 3]
    assert Itr([2, 3]).prod(10) == 60
    assert Itr[int]([]).prod(10) == 10


def test_find_map() -> None:
    def parse(s: str) -> int | None:
        return int(s) if s.isdigit() else None

    assert Itr(["a", "12", "3"]).find_map(parse) == 12
    assert Itr(["a", "b"]).find_map(parse) is None
    assert Itr[str]([]).find_map(parse) is None
    # short-circuits, so is safe on an infinite iterator
    assert Itr(itertools.count()).find_map(lambda n: n * n if n > 3 else None) == 16


def test_min_max() -> None:
    assert Itr([3, 1, 4, 1, 5, 9, 2, 6]).min_max() == (1, 9)
    assert Itr([7]).min_max() == (7, 7)
    assert Itr(["bb", "a", "ccc", "dd"]).min_max(len) == ("a", "ccc")


def test_min_max_ties_match_builtins() -> None:
    data = [(1, "a"), (0, "b"), (0, "c"), (2, "d"), (2, "e")]

    def key(t: tuple[int, str]) -> int:
        return t[0]

    assert Itr(data).min_max(key) == (min(data, key=key), max(data, key=key)) == ((0, "b"), (2, "d"))


def test_min_max_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        Itr[int]([]).min_max()
