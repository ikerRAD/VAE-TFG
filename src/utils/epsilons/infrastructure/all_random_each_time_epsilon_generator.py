import tensorflow as tf

from src.utils.epsilons.domain.epsilon_generator import EpsilonGenerator

"""
This epsilon generator will always generate the same epsilon set. All the rows could be different and the value
of the set will change everytime the get method is called.
"""


class AllRandomEachTimeEpsilonGenerator(EpsilonGenerator):
    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return tf.random.normal((self._rows, self._cols))
