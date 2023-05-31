from enum import Enum
from typing import Union, List

from src.utils.batches.infrastructure.random_strict_batch import RandomStrictBatch
from src.utils.batches.infrastructure.random_batch import RandomBatch
from src.utils.batches.infrastructure.cyclic_batch import CyclicBatch
from src.utils.batches.infrastructure.strict_batch import StrictBatch
from src.utils.batches.infrastructure.common_batch import CommonBatch
from src.utils.batches.domain.batch import Batch

"""
Class for selecting a batch given the batch itself or an identification string.
"""


class BatchSelector:
    class Batches(Enum):
        COMMON_BATCH = "common"
        STRICT_BATCH = "strict"
        CYCLIC_BATCH = "cyclic"
        RANDOM_BATCH = "random"
        RANDOM_STRICT_BATCH = "random_strict"

    @classmethod
    def possible_keys(cls) -> List[str]:
        return [elem.value for elem in cls.Batches]

    @classmethod
    def select(cls, batch_type: Union[str, Batch]) -> Batch:
        if batch_type == cls.Batches.COMMON_BATCH:
            return CommonBatch()

        if batch_type == cls.Batches.STRICT_BATCH:
            return StrictBatch()

        if batch_type == cls.Batches.CYCLIC_BATCH:
            return CyclicBatch()

        if batch_type == cls.Batches.RANDOM_BATCH:
            return RandomBatch()

        if batch_type == cls.Batches.RANDOM_STRICT_BATCH:
            return RandomStrictBatch()

        return batch_type
