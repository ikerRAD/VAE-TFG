from typing import Optional

import tensorflow as tf

from src.project.domain.Exceptions.no_more_batches_exception import (
    NoMoreBatchesException,
)
from src.utils.batches.domain.batch import Batch
from src.utils.batches.domain.slice import Slice

"""
Common batch, divides the dataset in batches of the specified size. The last division might be smaller than the rest.
When there is no next batch an exception is raised.
"""


class CommonBatch(Batch):
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

        if self.__index_from >= self.__dataset.shape[0]:
            raise NoMoreBatchesException

        slice_to_return: Slice = Slice(
            data=self.__dataset[self.__index_from : self.__index_to]
        )

        self.__index_from = self.__index_to
        self.__index_to = self.__index_to + self.__batch_size

        return slice_to_return.data
