from abc import ABC, abstractmethod

import tensorflow as tf

"""
Generic skeleton to build up new batches
"""


class Batch(ABC):
    @abstractmethod
    def set_up(self, dataset: tf.Tensor, size: int) -> None:
        pass

    @abstractmethod
    def next(self) -> tf.Tensor:
        pass
