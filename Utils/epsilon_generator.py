from abc import ABC, abstractmethod
from typing import Optional, List

import tensorflow as tf

"""
Generic skeleton to build up new epsilon generators
"""


class EpsilonGenerator(ABC):
    def __init__(self) -> None:
        self._cols: Optional[int] = None
        self._cols: Optional[int] = None

    def set_up(self, n_rows: int, n_cols: int) -> None:
        assert n_rows > 0
        assert n_cols > 0
        self._rows = n_rows
        self._cols = n_cols

    @abstractmethod
    def get(self) -> tf.Tensor:
        pass


"""
This epsilon generator will always generate the same epsilon. All the rows will be the same and the value
will always be the same no matter how many different executions are made.
"""


class AlwaysSameEpsilonGenerator(EpsilonGenerator):
    def __init__(self) -> None:
        super().__init__()
        self._epsilon: Optional[tf.Tensor] = None

    def set_up(self, n_rows: int, n_cols: int) -> None:
        super().set_up(n_rows, n_cols)
        one_epsilon: List[tf.Tensor] = [tf.random.normal((1, self._cols), seed=111)]
        self._epsilon = tf.convert_to_tensor(one_epsilon * self._rows)

    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return self._epsilon


"""
This epsilon generator will always generate the same epsilon set. All the rows could be different but the value
of the set will always be the same no matter how many different executions are made.
"""


class AlwaysSameEpsilonSetGenerator(EpsilonGenerator):
    def __init__(self) -> None:
        super().__init__()
        self._epsilon: Optional[tf.Tensor] = None

    def set_up(self, n_rows: int, n_cols: int) -> None:
        super().set_up(n_rows, n_cols)
        self._epsilon = tf.random.normal((self._rows, self._cols), seed=111)

    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return self._epsilon


"""
This epsilon generator will generate the same epsilon. All the rows will be the same but the value
will change with every different execution.
"""


class SameEpsilonGenerator(EpsilonGenerator):
    def __init__(self) -> None:
        super().__init__()
        self._epsilon: Optional[tf.Tensor] = None

    def set_up(self, n_rows: int, n_cols: int) -> None:
        super().set_up(n_rows, n_cols)
        one_epsilon: List[tf.Tensor] = [tf.random.normal((1, self._cols))]
        self._epsilon = tf.convert_to_tensor(one_epsilon * self._rows)

    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return self._epsilon


"""
This epsilon generator will always generate the same epsilon set. All the rows could be different and the value
of the set will change among executions.
"""


class SameEpsilonSetGenerator(EpsilonGenerator):
    def __init__(self) -> None:
        super().__init__()
        self._epsilon: Optional[tf.Tensor] = None

    def set_up(self, n_rows: int, n_cols: int) -> None:
        super().set_up(n_rows, n_cols)
        self._epsilon = tf.random.normal((self._rows, self._cols))

    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return self._epsilon


"""
This epsilon generator will generate the same epsilon. All the rows will be the same but the value
will change everytime the get method is called.
"""


class OnlyOneEachTimeEpsilonGenerator(EpsilonGenerator):
    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return tf.convert_to_tensor([tf.random.normal((1, self._cols))] * self._rows)


"""
This epsilon generator will always generate the same epsilon set. All the rows could be different and the value
of the set will change everytime the get method is called.
"""


class AllRandomEachTimeEpsilonGenerator(EpsilonGenerator):
    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return tf.random.normal((self._rows, self._cols))
