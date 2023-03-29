from typing import Optional

import tensorflow as tf

from utils.epsilons.domain.epsilon_generator import EpsilonGenerator

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
