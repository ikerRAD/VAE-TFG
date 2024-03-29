from typing import Optional, List

import tensorflow as tf

from src.utils.epsilons.domain.epsilon_generator import EpsilonGenerator

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
        one_epsilon: List[float] = tf.random.normal((1, self._cols)).numpy().tolist()
        self._epsilon = tf.convert_to_tensor(one_epsilon * self._rows)

    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return self._epsilon
