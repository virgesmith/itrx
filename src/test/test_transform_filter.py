
import pytest

from itrx import Itr


def test_filter() -> None:
    it = Itr([1, 2, 3, 4]).filter(lambda x: x % 2 == 0)
    assert it.collect() == (2, 4)


def test_flatten() -> None:
    it = Itr([[1, 2], [3], [], [4, 5]])
    assert it.flatten().collect() == (1, 2, 3, 4, 5)


def test_flat_map() -> None:
    it: Itr[int] = Itr([[1, 2], [3]]).flat_map(lambda x: x * 10)  # type: ignore[operator]
    assert it.collect() == (10, 20, 30)


def test_map() -> None:
    it = Itr([1, 2, 3]).map(lambda x: x * 2)
    assert it.collect() == (2, 4, 6)


def test_map_while() -> None:
    it = Itr(range(10)).map_while(lambda x: x < 5, lambda x: x * x)
    assert it.collect() == (0, 1, 4, 9, 16)


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

