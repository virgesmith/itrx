# itrx - A Chainable Iterable Adaptor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https://raw.githubusercontent.com/virgesmith/itrx/refs/heads/main/pyproject.toml)
![PyPI - Version](https://img.shields.io/pypi/v/itrx)


`itrx` is a Python library that adapts iterators, iterables, and generators, providing a Rust-inspired `Iterator` trait experience with added Pythonic conveniences. It enables developers to build complex data processing pipelines with a fluent, chainable, and lazy API. In most cases, it simply wraps itertools in syntactic sugar.

Heavily inspired by Rust's [Iterator trait](https://doc.rust-lang.org/std/iter/trait.Iterator.html), `itrx` offers a familiar and robust pattern for sequence manipulation.

## Key Features & Why `Itr`?

*   **Compatibility:** `Itr` is designed to work seamlessly with various Python iterable types, including sequences, ranges, custom iterators, and generators.
*   **Fluent & Chainable API:** Write expressive, left-to-right sequences of operations on your data.
*   **Lazy Evaluation:** Operations are only performed when their results are needed, improving efficiency for large or infinite sequences.
*   **Rust-Inspired Patterns:** Bring the power and clarity of Rust's `Iterator` design to your Python projects.
*   **Pythonic Additions:** Includes convenient methods like `starmap` that integrate smoothly with Python's ecosystem.
*   **Simplifies Complex Logic:** Often provides a more readable and concise alternative to nested `itertools` or built-in functions.


## Installation

```sh
pip install itrx
```

## Quick Start & Core Functionality

### Chaining

`Itr` wraps Python Iterators, Iterables, and Generators, allowing efficient chaining methods for data transformation.
In this completely arbitrary example we take some integers, reverse them, discard some, take every 4th value and
print it if its square ends with the digit 9:

```py
>>> from itrx import Itr
>>> Itr(range(100)).rev().step_by(4).skip(10).map(lambda x: x * x).filter(lambda x: x % 10 == 9).for_each(print)
2209
1849
729
529
49
9

```

For reference, the equivalent expression using built-ins and `itertools` can be significantly less readable:

```python
from itertools import islice

for item in filter(
    lambda x: x % 10 == 9,
    map(lambda x: x * x, islice(islice(reversed(range(100)), None, None, 4), 10, None)),
):
    print(item)

```

### Outputs

While `Itr` methods typically return `self` or another `Itr` instance, the `collect()` method allows you to materialize the results into various Python collections: `tuple` (default), `list`, `set`, or `dict`.

Here's how to group words by their length into a dictionary:

```python
>>> from itrx import Itr
>>> Itr(("apple", "banana", "carrot")).groupby(len).collect(dict)
{5: ('apple',), 6: ('banana', 'carrot')}

```

For reference, an equivalent using `itertools` directly:

```py
>>> import itertools
>>> {k: tuple(v) for k, v in itertools.groupby(("apple", "banana", "carrot"), key=len)}
{5: ('apple',), 6: ('banana', 'carrot')}

```

*Note: Using `collect(dict)` requires an iterable that produces 2-tuples (key-value pairs).*


## How `Itr` Works: Lazy vs. Eager

Most `Itr` methods are **lazy transformations**, meaning they return a new `Itr` instance without immediately processing any data. This allows for arbitrary chaining and efficient memory usage, as items are only processed as they are requested. In most cases, `Itr` simply acts as a convenient wrapper around `itertools`, enabling this left-to-right chaining syntax.

- **Combining and splitting:**  `partition`, `copy`, `batched`, `pairwise`, `rolling`, `chain`, `cycle`, `repeat`, `product`, `inspect`
- **Transformation and filtering:** `filter`, `map`, `starmap`, `map_while`, `flatten`, `flat_map`, `skip_while`, `take_while`, `groupby`

However, some methods are **eager consumers**. These methods iterate over and consume the underlying data, returning concrete values, collections, or aggregates. Examples include:

*   **Collection methods:** `collect`, `next`, `next_chunk`, `nth`, `position`
*   **Aggregation methods:** `count`, `reduce`, `max`, `min`, `all`, `any`, `last`, `find`, `fold`

### Important Considerations

When working with `Itr`, keep these points in mind:

*   **Single-Pass Iterators:** Like all Python iterators, `Itr` instances (and their underlying iterators) can generally only be consumed once. If you need to process the same sequence multiple times, use methods like `copy()`, `cycle()`, or `repeat()` as necessary.
*   **No Rewinding:** It's not possible to rewind an `Itr` to an earlier state. You can "preview" the next value using the `peek()` method, but be aware that `peek()` often copies the iterator internally, which can be inefficient if used excessively.
*   **Infinite Iterators:** Be cautious with open-ended iterators (e.g., those from `itertools.count()` or custom generators). Eager evaluation methods (like `collect()`, `count()`, `reduce()`) will attempt to consume the entire sequence, potentially leading to infinite loops or out-of-memory errors if applied to an infinite source.

## API Reference

`Itr` provides a comprehensive set of methods for various iterable operations. For a complete list of methods and their detailed descriptions, please refer to the [API documentation](./doc/apidoc.md).

*Note: `apidoc.md` is auto-generated using `Itr` - see [introspect.py](src/scripts/introspect.py).*

## Examples

Some worked examples can be found [in this notebook](./doc/examples.ipynb).
