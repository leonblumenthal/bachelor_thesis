import os
from abc import ABC, abstractmethod
from typing import Any, Iterator, List


class Loader(ABC):
    """Abstract helper class to load sorted items from a directory by index"""

    def __init__(self, dir_path: str, ending: str):
        self.paths = [
            os.path.join(dir_path, name)
            for name in sorted(os.listdir(dir_path))
            if name.endswith(ending)
        ]

    def load_item(self, index: int, **kwargs) -> Any:
        """Load a single item by index."""
        path = self.paths[index]

        item = self._load_item(path, **kwargs)

        return item

    def load_items(self, indices: List[int] = None, **kwargs) -> Iterator[Any]:
        """Lazily load (all) items."""

        if indices is None:
            indices = range(len(self.paths))

        for index in indices:
            yield self.load_item(index, **kwargs)

    @classmethod
    @abstractmethod
    def _load_item(cls, path: str, **kwargs) -> Any:
        """Load item from path."""
        pass
