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
