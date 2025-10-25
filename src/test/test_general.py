from collections.abc import Iterator

import pytest

from itrx import Itr


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


def test_exhaust_triggers_side_effects_and_consumes_all() -> None:
    side = []

    def gen() -> Iterator[int]:
        for v in (10, 20, 30):
            side.append(v)
            yield v

    itr = Itr(gen())
    assert side == []
    result = itr.exhaust()  # type: ignore[func-returns-value]
    assert result is None
    assert side == [10, 20, 30]
    # iterator should now be exhausted
    assert itr.collect() == ()


def test_exhaust_consumes_remaining_only() -> None:
    itr = Itr(iter([1, 2, 3, 4]))
    first = itr.next()
    assert first == 1
    itr.exhaust()
    # remaining items were consumed, collect yields empty tuple
    assert itr.collect() == ()


def test_exhaust_on_empty_iterator_no_error() -> None:
    itr: Itr[None] = Itr(())
    itr.exhaust()  # should not raise
    assert itr.collect() == ()
