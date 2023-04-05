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
    @staticmethod
    def possible_keys() -> List[str]:
        return [
            "common",
            "strict",
            "cyclic",
            "random",
            "random_strict",
        ]

    @staticmethod
    def select(batch_type: Union[str, Batch]) -> Batch:
        if batch_type == "common":
            return CommonBatch()
        elif batch_type == "strict":
            return StrictBatch()
        elif batch_type == "cyclic":
            return CyclicBatch()
        elif batch_type == "random":
            return RandomBatch()
        elif batch_type == "random_strict":
            return RandomStrictBatch()
        else:
            return batch_type
