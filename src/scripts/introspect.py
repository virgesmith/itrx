from operator import add

from itrx import Itr
from itrx import __version__ as itrx_version

# use Itr to introspect itself for methods and their docstrings
methods = (
    Itr(dir(Itr))
    .filter(lambda m: m in ("__init__", "__iter__", "__next__") or not m.startswith("_"))
    .map(lambda m: (m, getattr(Itr, m).__doc__))
    .collect(dict)
)

# use Itr to make some documentation
method_template = """
### `{method_name}`

{method_doc}
"""

with open("./doc/apidoc.md", "w") as fd:
    fd.write(f"# `Itr` v{itrx_version} class documentation\n")
    fd.write(Itr.__doc__ or "")
    fd.write("## Public methods\n")
    fd.write(Itr(methods.items()).map(lambda m: method_template.format(method_name=m[0], method_doc=m[1])).reduce(add))
