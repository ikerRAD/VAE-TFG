from typing import Optional, List

import numpy as np
import tensorflow as tf

from utils.batches.domain.batch import Batch
from utils.batches.domain.slice import Slice


"""
Random batch, divides the dataset in batches of the specified size. The last division might be smaller than the rest.
It is never known which batch will be chosen as next, it is random.
"""


class RandomBatch(Batch):
    def __init__(self) -> None:
        self.__batches: Optional[List[tf.Tensor]] = None

    def set_up(self, dataset: tf.Tensor, size: int) -> None:
        assert dataset.shape[0] > size
        assert size > 0

        froms: List[int] = [0]
        tos: List[int] = [size]

        while tos[-1] <= dataset.shape[0]:
            froms.append(tos[-1])
            tos.append(tos[-1] + size)

        batches: List[Slice] = []
        for from_batch, to_batch in zip(froms, tos):
            batches.append(Slice(data=dataset[from_batch:to_batch]))
        self.__batches: np.ndarray = np.array(batches, dtype=object)

    def next(self) -> tf.Tensor:
        assert self.__batches is not None

        return np.random.choice(self.__batches).data
