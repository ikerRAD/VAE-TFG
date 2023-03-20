"""
Exception for when the given value is incorrect.
"""


class IllegalValueException(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message

    def __str__(self) -> str:
        return self.message
