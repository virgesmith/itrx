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
    with pytest.raises(ValueError):
        it.nth(-1)
    assert it.nth(0) == 10
    # consumes preceding items, so this advances from the current position
    assert it.nth(2) == 40
    with pytest.raises(StopIteration):
        it.nth(10)


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


def test_peek_repeated_is_stable() -> None:
    it = Itr([1, 2])
    assert it.peek() == 1
    assert it.peek() == 1
    assert it.next() == 1
    assert it.peek() == 2


def test_peek_does_not_deepen_iterator() -> None:
    # peek is O(1): it must not wrap the underlying iterator in a new layer on each call
    it = Itr([1, 2])
    for _ in range(10000):
        assert it.peek() == 1
    assert it.collect() == (1, 2)


def test_peek_handles_none_values() -> None:
    it = Itr([None, 1])
    assert it.peek() is None
    assert it.next() is None
    assert it.next() == 1


def test_peeked_item_seen_by_adaptors() -> None:
    # the buffered item must be pushed back before any adaptor consumes the underlying iterator
    it = Itr([1, 2, 3])
    assert it.peek() == 1
    assert it.map(lambda x: x * 10).collect() == (10, 20, 30)


def test_peeked_item_seen_by_copy() -> None:
    it = Itr([1, 2, 3])
    assert it.peek() == 1
    copied = it.copy()
    assert copied.collect() == (1, 2, 3)
    assert it.collect() == (1, 2, 3)


def test_peeked_item_seen_by_for_loop() -> None:
    it = Itr([1, 2, 3])
    assert it.peek() == 1
    assert list(it) == [1, 2, 3]


def test_unpeek_retains_pending_item() -> None:
    it = Itr([1, 2, 3])
    assert it.peek() == 1
    assert it.unpeek().collect() == (1, 2, 3)


def test_unpeek_with_empty_buffer() -> None:
    it = Itr([1, 2, 3])
    assert it.peek() == 1
    assert it.next() == 1  # drains the buffer
    assert it.unpeek().collect() == (2, 3)


def test_unpeek_without_peek_is_noop() -> None:
    it = Itr([1, 2, 3])
    assert it.unpeek().collect() == (1, 2, 3)


def test_peek_after_unpeek() -> None:
    it = Itr([1, 2, 3])
    assert it.peek() == 1
    assert it.unpeek().peek() == 1
    assert it.collect() == (1, 2, 3)


def test_next_if_consumes_on_match() -> None:
    it = Itr([1, 2, 3])
    assert it.next_if(lambda x: x < 3) == 1
    assert it.next_if(lambda x: x < 3) == 2
    assert it.next_if(lambda x: x < 3) is None
    # the non-matching item is not consumed
    assert it.next() == 3


def test_next_if_exhausted_returns_none() -> None:
    it = Itr[int]([])
    assert it.next_if(lambda _: True) is None


def test_next_if_after_peek() -> None:
    it = Itr([5, 6])
    assert it.peek() == 5
    assert it.next_if(lambda x: x == 5) == 5
    assert it.next_if(lambda x: x == 5) is None
    assert it.peek() == 6
