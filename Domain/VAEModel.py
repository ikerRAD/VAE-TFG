from abc import abstractmethod, ABC
from typing import List
import tensorflow as tf
from Utils.image import Image

"""
Interface for all the VAE and CVAE implementations. The interface follows the a
sklearn-like structure.
"""


class VAEModel(ABC, tf.keras.Model):

    @abstractmethod
    def fit_dataset(self, x: List[Image]) -> List[float]:  # TODO List[Images]
        pass

    @abstractmethod
    def generate_with_random_sample(self) -> Image:
        pass

    @abstractmethod
    def generate(self, sample: List[int]) -> Image:
        pass
