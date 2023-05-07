"""
Exception for when the batch has no more batches and the iterations have to stop.
"""


class NoMoreBatchesException(Exception):
    def __init__(self) -> None:
        self.message: str = "There are no more batches to iterate with."

    def __str__(self) -> str:
        return self.message
