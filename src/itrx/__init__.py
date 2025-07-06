import importlib.metadata

__version__ = importlib.metadata.version("itrx")

from .itr import Itr

__all__ = ["Itr"]
