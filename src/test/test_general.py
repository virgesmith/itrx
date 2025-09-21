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
