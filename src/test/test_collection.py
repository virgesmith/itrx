import pytest

from itrx import Itr


def test_collect() -> None:
    it = Itr([1, 2, 3])
    assert it.collect() == (1, 2, 3)

    # collect doesnt raise StopIteration
    assert it.collect() == ()


def test_collect_types() -> None:
    it = Itr([1, 2, 3])
    assert it.collect(list) == [1, 2, 3]

    it = Itr([1, 2, 3])
    assert it.collect(set) == {1, 2, 3}

    it = Itr([1, 2, 3])
    with pytest.raises(TypeError):
        it.collect(dict)

    it2 = Itr("abc").zip((1, 2, 3))
    assert it2.collect(dict) == {"a": 1, "b": 2, "c": 3}


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


def test_position() -> None:
    assert Itr("abcdefghijklmnopqrstuvwxyz").position(lambda x: x == "a") == 0
    with pytest.raises(StopIteration):
        Itr("abcdefghijklmnopqrstuvwxyz").position(lambda x: x == "H")

    a = Itr("abcdefghijklmnopqrstuvwxyz")
    assert a.position(lambda x: x == "h") == 7
    assert a.position(lambda x: x == "z") == 17  # counter is reset
    with pytest.raises(StopIteration):
        assert a.position(lambda x: x == "a")  # iterator is exhausted


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
