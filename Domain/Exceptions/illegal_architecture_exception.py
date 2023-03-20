"""
Exception for when the given architecture is not possible.
"""


class IllegalArchitectureException(Exception):
    def __init__(self, message: str) -> None:
        self.message: str = message

    def __str__(self) -> str:
        return self.message
