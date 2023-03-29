from abc import ABC, abstractmethod
from typing import Optional

import tensorflow as tf

"""
Generic skeleton to build up new epsilon generators
"""


class EpsilonGenerator(ABC):
    def __init__(self) -> None:
        self._cols: Optional[int] = None
        self._cols: Optional[int] = None

    def set_up(self, n_rows: int, n_cols: int) -> None:
        assert n_rows > 0
        assert n_cols > 0
        self._rows = n_rows
        self._cols = n_cols

    @abstractmethod
    def get(self) -> tf.Tensor:
        pass
