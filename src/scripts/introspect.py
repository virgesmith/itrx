from operator import add
from pathlib import Path

from itrx import Itr
from itrx import __version__ as itrx_version

method_template = """
### `{method_name}`

{method_doc}
"""


def generate_apidoc(cls: type, file: Path) -> None:
    """
    Use Itr to introspect itself for methods and their docstrings and write them to a markdown file
    """
    methods = (
        Itr(dir(cls))
        .filter(lambda m: m in ("__init__", "__iter__", "__next__") or not m.startswith("_"))
        .map(lambda m: (m, getattr(Itr, m).__doc__))
    )

    with open(file, "w") as fd:
        fd.write(f"# `Itr` v{itrx_version} class documentation\n")
        fd.write(Itr.__doc__ or "")
        fd.write("## Public methods\n")
        fd.write(methods.map(lambda m: method_template.format(method_name=m[0], method_doc=m[1])).reduce(add))


if __name__ == "__main__":
    generate_apidoc(Itr, Path("./doc/apidoc.md"))
