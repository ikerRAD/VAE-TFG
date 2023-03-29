from typing import Optional

import tensorflow as tf

from utils.batches.domain.batch import Batch
from utils.batches.domain.slice import Slice

"""
Cyclic batch, divides the dataset in batches of the specified size. It will always have a next batch.
When the end of the dataset arrives, treats it as a cyclic data structure.
"""


class CyclicBatch(Batch):
    def __init__(self) -> None:
        self.__dataset: Optional[tf.Tensor] = None
        self.__index_from: Optional[int] = None
        self.__index_to: Optional[int] = None
        self.__batch_size: Optional[int] = None

    def set_up(self, dataset: tf.Tensor, size: int) -> None:
        assert dataset.shape[0] > size
        assert size > 0

        self.__dataset = dataset
        self.__index_from = 0
        self.__index_to = size
        self.__batch_size = size

    def next(self) -> tf.Tensor:
        assert self.__dataset is not None

        batch_to_return: Slice
        if self.__index_to > self.__dataset.shape[0]:
            from1: int = self.__index_from
            to1: int = self.__dataset.shape[0]

            self.__index_from = 0
            self.__index_to = self.__index_to - self.__dataset.shape[0]

            batch_to_return = Slice(
                data=tf.concat(
                    [
                        self.__dataset[from1:to1],
                        self.__dataset[self.__index_from : self.__index_to],
                    ],
                    axis=0,
                )
            )
        else:
            batch_to_return = Slice(
                data=self.__dataset[self.__index_from : self.__index_to]
            )

        self.__index_from = self.__index_to
        self.__index_to = self.__index_to + self.__batch_size

        return batch_to_return.data
