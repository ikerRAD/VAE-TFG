from abc import abstractmethod, ABC
from typing import Optional, Union, List, Tuple, Dict
import tensorflow as tf
from src.project.domain.exceptions.illegal_architecture_exception import (
    IllegalArchitectureException,
)
from src.project.domain.exceptions.illegal_value_exception import IllegalValueException
from src.utils.batches.domain.batch import Batch
from src.utils.epsilons.domain.epsilon_generator import EpsilonGenerator

"""
Interface for all the VAE and CVAE implementations. The interface follows the a
sklearn-like structure.
"""


class VAEModel(ABC, tf.keras.Model):
    def __init__(
        self, learning_rate: float, n_distributions: int, max_iter: int, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__do_hyperparam_checks(learning_rate, n_distributions, max_iter)

        self._latent: int = n_distributions
        self._learning_rate: float = learning_rate
        self._optimizer = tf.keras.optimizers.Adam(self._learning_rate)
        self._iterations: int = max_iter

    @abstractmethod
    def fit_dataset(
        self,
        return_loss: bool,
        epsilon_generator: Union[str, EpsilonGenerator],
        batch_size: int,
        batch_type: Optional[Union[str, Batch]],
        generate_samples: bool,
        sample_frequency: int,
    ) -> Optional[Tuple[List[float], List[Dict[str, float]]]]:
        pass

    @abstractmethod
    def _encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    @abstractmethod
    def _reparameterize(self, means: tf.Tensor, logvars: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def _decode(self, z: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def encode_and_decode(self, x: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def change_dataset(
        self,
        dataset: tf.Tensor,
        normalize_data: bool,
        discretize_data: bool,
    ) -> None:
        pass

    @abstractmethod
    def add_train_instances(
        self,
        instances: tf.Tensor,
        normalize_data: bool,
        discretize_data: bool,
        shuffle_data: bool,
    ) -> None:
        pass

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def get_latent_space(self) -> int:
        return self._latent

    def get_max_iterations(self) -> int:
        return self._iterations

    def set_max_iterations(self, max_iter: int) -> None:
        if max_iter <= 0:
            raise IllegalValueException(
                "The maximum number of iterations cannot be less than or equal to zero."
            )
        self._iterations = max_iter

    def _random_sample(self, n_samples: int = 1) -> tf.Tensor:
        return tf.random.normal(shape=(n_samples, self._latent))

    def generate_with_random_sample(self, n_samples: int = 1) -> tf.Tensor:
        sample: tf.Tensor = self._random_sample(n_samples)
        return self._decode(sample)

    def generate_with_one_sample(
        self, sample: List[float], n_samples: int = 1
    ) -> tf.Tensor:
        samples: tf.Tensor = tf.convert_to_tensor(
            [sample for _ in range(n_samples)], tf.float32
        )
        return self._decode(samples)

    def generate_with_multiple_samples(self, samples: tf.Tensor) -> tf.Tensor:
        return self._decode(samples)

    @staticmethod
    def __do_hyperparam_checks(
        learning_rate: float,
        n_distributions: int,
        max_iter: int,
    ) -> None:
        if learning_rate <= 0:
            raise IllegalValueException(
                "The 'learning rate' parameter cannot be less than or equal to zero."
            )

        if max_iter <= 0:
            raise IllegalValueException(
                "The maximum number of iterations cannot be less than or equal to zero."
            )

        if n_distributions <= 0:
            raise IllegalArchitectureException(
                "The number of distributions cannot be less than or equal to zero."
            )
