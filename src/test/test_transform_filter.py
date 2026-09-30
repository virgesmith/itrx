import itertools
from collections import defaultdict
from operator import mul

import pytest

from itrx import Itr


def test_accumulate_default_sum() -> None:
    result = Itr([1, 2, 3]).accumulate().collect()
    assert result == (1, 3, 6)


def test_accumulate_with_func() -> None:
    result = Itr([2, 3, 4]).accumulate(lambda x, y: x * y).collect()
    assert result == (2, 6, 24)

    result = Itr([2, 3, 4]).accumulate(mul).collect()
    assert result == (2, 6, 24)


def test_accumulate_with_initial() -> None:
    result = Itr([1, 2, 3]).accumulate(mul, initial=10).collect()
    assert result == (10, 10, 20, 60)


def test_accumulate_empty() -> None:
    data: list[int] = []
    result = Itr(data).accumulate().collect()
    assert result == ()


def test_accumulate_single_element() -> None:
    result = Itr([5]).accumulate().collect()
    assert result == (5,)


def test_filter() -> None:
    it = Itr([1, 2, 3, 4]).filter(lambda x: x % 2 == 0)
    assert it.collect() == (2, 4)


def test_flatten() -> None:
    it = Itr([[1, 2], [3], [], [4, 5]])
    assert it.flatten().collect() == (1, 2, 3, 4, 5)


def test_flat_map() -> None:
    it = Itr([1, 2, 3]).flat_map(lambda n: [n] * n)
    assert it.collect() == (1, 2, 2, 3, 3, 3)

    it = Itr([1, 2, 3]).flat_map(lambda n: range(n))
    assert it.collect() == (0, 0, 1, 0, 1, 2)


def test_flat_map_empty() -> None:
    it: Itr[int] = Itr([]).flat_map(lambda n: [n] * n)
    assert it.collect() == ()


def test_flat_map_invalid_mapper() -> None:
    # mapper must return an iterable
    with pytest.raises(TypeError):
        Itr([1, 2, 3]).flat_map(lambda n: n * 2).collect()  # ty: ignore[invalid-argument-type]


def test_map() -> None:
    it = Itr([1, 2, 3]).map(lambda x: x * 2)
    assert it.collect() == (2, 4, 6)


def test_map_while() -> None:
    it = Itr(range(10)).map_while(lambda x: x < 5, lambda x: x * x)
    assert it.collect() == (0, 1, 4, 9, 16)


def test_map_dict_basic() -> None:
    mapper = {1: "a", 2: "b", 3: "c"}
    assert Itr([1, 2, 3]).map_dict(mapper).collect() == ("a", "b", "c")


def test_map_dict_defaultdict_handles_missing_keys() -> None:
    mapper = defaultdict(lambda: "x", {1: "a"})
    assert Itr([1, 2, 99]).map_dict(mapper).collect() == ("a", "x", "x")
    # defaultdict should have created entries for missing keys when accessed
    assert 2 in mapper and 99 in mapper


def test_map_dict_missing_key_raises_keyerror_for_plain_dict() -> None:
    mapper = {1: "a"}
    with pytest.raises(KeyError):
        Itr([1, 2]).map_dict(mapper).collect()


def test_map_dict_empty_iterable_returns_empty_tuple() -> None:
    mapper = {1: "a"}
    empty: list[int] = []
    assert Itr(empty).map_dict(mapper).collect() == ()


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
    it = Itr[str]([]).skip_while(lambda _: True)
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
    taken = it.take_while(lambda _: True)
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


def test_groupby_unsorted() -> None:
    it = Itr(["bb", "a", "ccc", "dd", "e"]).groupby(len, sort=False)
    assert it.collect() == ((2, ("bb", "dd")), (1, ("a", "e")), (3, ("ccc",)))


def test_groupby_unsorted_needs_only_hashable_keys() -> None:
    # None and int keys can't be sorted together, but can be hashed
    data = [1, None, 2, None]
    with pytest.raises(TypeError):
        Itr(data).groupby(lambda x: x)
    assert Itr(data).groupby(lambda x: x, sort=False).collect(dict) == {1: (1,), None: (None, None), 2: (2,)}


def test_chunk_by() -> None:
    # consecutive runs only, order preserved, no sorting (unlike groupby)
    it = Itr([1, 1, 2, 3, 3, 1]).chunk_by(lambda x: x)
    assert it.collect() == ((1, (1, 1)), (2, (2,)), (3, (3, 3)), (1, (1,)))


def test_chunk_by_lazy_on_infinite() -> None:
    # chunk_by is lazy, so it works on unbounded iterators
    counts = Itr(itertools.count()).chunk_by(lambda n: n // 2).take(3).collect()
    assert counts == ((0, (0, 1)), (1, (2, 3)), (2, (4, 5)))


def test_dedup() -> None:
    assert Itr([1, 1, 2, 2, 2, 3, 1]).dedup().collect() == (1, 2, 3, 1)


def test_dedup_no_duplicates() -> None:
    assert Itr([1, 2, 3]).dedup().collect() == (1, 2, 3)


def test_dedup_empty() -> None:
    assert Itr[int]([]).dedup().collect() == ()


def test_dedup_unhashable_items() -> None:
    assert Itr([[1], [1], [2]]).dedup().collect() == ([1], [2])


def test_dedup_lazy_on_infinite() -> None:
    it = Itr(itertools.count()).flat_map(lambda x: (x, x)).dedup()
    assert it.take(3).collect() == (0, 1, 2)


def test_dedup_with_count() -> None:
    assert Itr([4, 4, 2, 3, 3, 1]).dedup_with_count().collect() == ((4, 2), (2, 1), (3, 2), (1, 1))


def test_dedup_with_count_no_duplicates() -> None:
    assert Itr([1, 2, 3]).dedup_with_count().collect() == ((1, 1), (2, 1), (3, 1))


def test_dedup_with_count_empty() -> None:
    assert Itr[int]([]).dedup_with_count().collect() == ()


def test_dedup_with_count_single_run() -> None:
    assert Itr([7, 7, 7, 7]).dedup_with_count().collect() == ((7, 4),)


def test_dedup_with_count_unhashable_items() -> None:
    assert Itr([[1], [1], [2]]).dedup_with_count().collect() == (([1], 2), ([2], 1))


def test_dedup_with_count_lazy_on_infinite() -> None:
    it = Itr(itertools.count()).flat_map(lambda x: (x, x)).dedup_with_count()
    assert it.take(3).collect() == ((0, 2), (1, 2), (2, 2))


def test_dedup_with_count_keys_match_dedup() -> None:
    data = [1, 1, 2, 2, 2, 3, 1]
    counted = Itr(data).dedup_with_count().starmap(lambda item, _n: item).collect()
    assert counted == Itr(data).dedup().collect()


def test_dedup_with_count_differs_from_value_counts() -> None:
    # dedup_with_count counts adjacent runs positionally, value_counts counts occurrences overall
    data = [4, 4, 2, 3, 3, 1]
    assert Itr(data).dedup_with_count().collect() == ((4, 2), (2, 1), (3, 2), (1, 1))
    assert Itr(data).value_counts().collect() == ((4, 2), (3, 2), (2, 1), (1, 1))


def test_sorted_by() -> None:
    assert Itr(["ccc", "a", "bb"]).sorted_by(len).collect() == ("a", "bb", "ccc")


def test_sorted_by_reverse() -> None:
    assert Itr([1, 3, 2]).sorted_by(lambda x: x, reverse=True).collect() == (3, 2, 1)


def test_sorted_by_is_stable() -> None:
    data = [(1, "b"), (0, "a"), (1, "a"), (0, "b")]
    assert Itr(data).sorted_by(lambda x: x[0]).collect() == ((0, "a"), (0, "b"), (1, "b"), (1, "a"))


def test_sorted_by_empty() -> None:
    assert Itr[int]([]).sorted_by(lambda x: x).collect() == ()


def test_scan() -> None:
    assert Itr([1, 2, 3, 4]).scan(0, lambda total, x: (total + x, total + x)).collect() == (1, 3, 6, 10)


def test_scan_stops_on_none() -> None:
    result = Itr([1, 2, 3, 4]).scan(0, lambda total, x: None if total + x > 5 else (total + x, total + x))
    assert result.collect() == (1, 3)


def test_scan_empty() -> None:
    assert Itr[int]([]).scan(0, lambda total, x: (total + x, total + x)).collect() == ()


def test_scan_state_type_differs_from_output() -> None:
    # state is a running count, output is the item tagged with its 1-based position
    result = Itr("abc").scan(0, lambda n, c: (n + 1, f"{n + 1}{c}"))
    assert result.collect() == ("1a", "2b", "3c")


def test_scan_can_yield_none_outputs() -> None:
    # None is only a halt signal as the whole return value, not as an output
    assert Itr([1, 2]).scan(0, lambda total, x: (total + x, None)).collect() == (None, None)


def test_scan_lazy_on_infinite() -> None:
    running = Itr(itertools.count(1)).scan(0, lambda total, x: (total + x, total + x))
    assert running.take(4).collect() == (1, 3, 6, 10)


def test_sorted_by_no_key() -> None:
    assert Itr([3, 1, 2]).sorted_by().collect() == (1, 2, 3)
    assert Itr("cab").sorted_by(reverse=True).collect() == ("c", "b", "a")


def test_filter_map() -> None:
    def parse(s: str) -> int | None:
        return int(s) if s.isdigit() else None

    # only None is dropped, not other falsy values
    assert Itr(["1", "x", "0", ""]).filter_map(parse).collect() == (1, 0)
    assert Itr[str]([]).filter_map(parse).collect() == ()
    assert Itr(itertools.count()).filter_map(lambda n: n if n % 3 == 0 else None).take(3).collect() == (0, 3, 6)


def test_flat_map_lazy() -> None:
    assert Itr([1, 2]).flat_map(lambda n: [n] * n).collect() == (1, 2, 2)
    assert Itr(itertools.count()).flat_map(lambda n: (n, -n)).take(4).collect() == (0, 0, 1, -1)


def test_unique() -> None:
    assert Itr([3, 1, 3, 2, 1]).unique().collect() == (3, 1, 2)
    assert Itr[int]([]).unique().collect() == ()
    assert Itr(["apple", "avocado", "banana", "blueberry"]).unique(lambda s: s[0]).collect() == ("apple", "banana")


def test_unique_lazy_on_infinite() -> None:
    assert Itr(itertools.count()).map(lambda n: n // 3).unique().take(4).collect() == (0, 1, 2, 3)


def test_unique_unhashable() -> None:
    with pytest.raises(TypeError):
        Itr([[1], [1]]).unique().collect()
    assert Itr([[1], [2], [1]]).unique(tuple).collect() == ([1], [2])


def test_enumerate() -> None:
    assert Itr("abc").enumerate().collect() == ((0, "a"), (1, "b"), (2, "c"))
    assert Itr("ab").enumerate(start=5).collect() == ((5, "a"), (6, "b"))
    assert Itr[str]([]).enumerate().collect() == ()
