from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
from numpy import ndarray

from Domain.Exceptions.no_more_batches_exception import NoMoreBatchesException


"""
Generic skeleton to build up new batches
"""


class Batch(ABC):
    @abstractmethod
    def set_up(self, dataset: ndarray, size: int) -> None:
        pass

    @abstractmethod
    def next(self) -> ndarray:
        pass


"""
Common batch, divides the dataset in batches of the specified size. The last division might be smaller than the rest.
When there is no next batch an exception is raised.
"""


class CommonBatch(Batch):
    def __init__(self) -> None:
        self.__dataset: Optional[ndarray] = None
        self.__index_from: Optional[int] = None
        self.__index_to: Optional[int] = None
        self.__batch_size: Optional[int] = None

    def set_up(self, dataset: ndarray, size: int) -> None:
        assert dataset.shape[0] > size
        assert size > 0

        self.__dataset = dataset
        self.__index_from = 0
        self.__index_to = size
        self.__batch_size = size
    def next(self) -> ndarray:
        assert self.__dataset is not None

        if self.__index_from >= self.__dataset.shape[0]:
            raise NoMoreBatchesException

        batch_to_return: ndarray = self.__dataset[
            self.__index_from : self.__index_to
        ]

        self.__index_from = self.__index_to
        self.__index_to = self.__index_to + self.__batch_size

        return batch_to_return


"""
Strict batch, divides the dataset in batches of the specified size. The last division might be ignored if it was meant
to be smaller. When there is no next batch an exception is raised.
"""


class StrictBatch(Batch):
    def __init__(self) -> None:
        self.__dataset: Optional[ndarray] = None
        self.__index_from: Optional[int] = None
        self.__index_to: Optional[int] = None
        self.__batch_size: Optional[int] = None

    def set_up(self, dataset: ndarray, size: int) -> None:
        assert dataset.shape[0] > size
        assert size > 0

        self.__dataset = dataset
        self.__index_from = 0
        self.__index_to = size
        self.__batch_size = size
    def next(self) -> ndarray:
        assert self.__dataset is not None

        if self.__index_to > self.__dataset.shape[0]:
            raise NoMoreBatchesException

        batch_to_return: ndarray = self.__dataset[
            self.__index_from : self.__index_to
        ]

        self.__index_from = self.__index_to
        self.__index_to = self.__index_to + self.__batch_size

        return batch_to_return


"""
Cyclic batch, divides the dataset in batches of the specified size. It will always have a next batch.
When the end of the dataset arrives, treats it as a cyclic data structure.
"""


class CyclicBatch(Batch):
    def __init__(self) -> None:
        self.__dataset: Optional[ndarray] = None
        self.__index_from: Optional[int] = None
        self.__index_to: Optional[int] = None
        self.__batch_size: Optional[int] = None

    def set_up(self, dataset: ndarray, size: int) -> None:
        assert dataset.shape[0] > size
        assert size > 0

        self.__dataset = dataset
        self.__index_from = 0
        self.__index_to = size
        self.__batch_size = size
    def next(self) -> ndarray:
        assert self.__dataset is not None

        batch_to_return: ndarray
        if self.__index_to > self.__dataset.shape[0]:
            from1: int = self.__index_from
            to1: int = self.__dataset.shape[0]

            self.__index_from = 0
            self.__index_to = self.__index_to - self.__dataset.shape[0]

            batch_to_return = np.append(
                self.__dataset[from1:to1],
                self.__dataset[self.__index_from : self.__index_to],
                axis=0,
            )
        else:
            batch_to_return = self.__dataset[self.__index_from : self.__index_to]

        self.__index_from = self.__index_to
        self.__index_to = self.__index_to + self.__batch_size

        return batch_to_return


"""
Random batch, divides the dataset in batches of the specified size. The last division might be smaller than the rest.
It is never known which batch will be chosen as next, it is random.
"""


class RandomBatch(Batch):
    def __init__(self) -> None:
        self.__batches: Optional[List[ndarray]] = None

    def set_up(self, dataset: ndarray, size: int) -> None:
        assert dataset.shape[0] > size
        assert size > 0

        froms: List[int] = [0]
        tos: List[int] = [size]

        while tos[-1] <= dataset.shape[0]:
            froms.append(tos[-1])
            tos.append(tos[-1] + size)

        self.__batches = []
        for from_batch, to_batch in zip(froms, tos):
            self.__batches.append(dataset[from_batch:to_batch])

    def next(self) -> ndarray:
        assert self.__batches is not None

        return np.random.choice(self.__batches)


"""
Random batch, divides the dataset in batches of the specified size. The last division might be ignored if it
was meant to be smaller than the rest. It is never known which batch will be chosen as next, it is random.
"""


class RandomStrictBatch(Batch):
    def __init__(self) -> None:
        self.__batches: Optional[List[ndarray]] = None

    def set_up(self, dataset: ndarray, size: int) -> None:
        assert dataset.shape[0] > size
        assert size > 0

        froms: List[int] = [0]
        tos: List[int] = [size]

        while tos[-1] + size <= dataset.shape[0]:
            froms.append(tos[-1])
            tos.append(tos[-1] + size)

        self.__batches = []
        for from_batch, to_batch in zip(froms, tos):
            self.__batches.append(dataset[from_batch:to_batch])

    def next(self) -> ndarray:
        assert self.__batches is not None

        return np.random.choice(self.__batches)
