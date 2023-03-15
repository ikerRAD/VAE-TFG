"""
Exception for whn the given value is incorrect.
"""


class IllegalValueException(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message

    def __str__(self) -> None:
        return self.message
