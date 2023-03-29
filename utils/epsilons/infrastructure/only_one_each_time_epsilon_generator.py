import tensorflow as tf

from utils.epsilons.domain.epsilon_generator import EpsilonGenerator

"""
This epsilon generator will generate the same epsilon. All the rows will be the same but the value
will change everytime the get method is called.
"""


class OnlyOneEachTimeEpsilonGenerator(EpsilonGenerator):
    def get(self) -> tf.Tensor:
        assert self._rows is not None
        assert self._cols is not None
        return tf.convert_to_tensor(
            tf.random.normal((1, self._cols)).numpy().tolist() * self._rows
        )
