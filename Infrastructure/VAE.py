from typing import List
import tensorflow as tf
import numpy as np
from Exceptions.illegal_architecture_exception import IllegalArchitectureException
from Domain.VAEModel import VAEModel
from Utils.image import Image

"""
Implementation of the most common version of the VAE.
"""


class VAE(VAEModel):

    def __init__(self, architecture_encoder: List[int], architecture_decoder: List[int], learning: float = 0.001,
                 n_distributions: int = 5, max_iter: int = -1, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        if len(architecture_encoder) == 0:
            raise IllegalArchitectureException("The architecture of the encoder cannot be empty")

        if len(architecture_decoder) == 0:
            raise IllegalArchitectureException("The architecture of the decoder cannot be empty")

        if architecture_encoder[-1] < n_distributions * 2:
            raise IllegalArchitectureException(
                f"The last layer of the encoder cannot be lower than {n_distributions * 2}")

        if architecture_decoder[0] < n_distributions:
            raise IllegalArchitectureException(
                f"The first layer of the decoder cannot be lower than {n_distributions}")

        self.__latent: int = n_distributions

        self.__encoder = tf.keras.Sequential()
        for n_neurons in architecture_encoder:
            self.__encoder.add(tf.keras.layers.Dense(n_neurons))
        self.__encoder.add(tf.keras.layers.Dense(self.__latent * 2))

        self.__decoder = tf.keras.Sequential()
        self.__decoder.add(tf.keras.layers.Dense(self.__latent))
        for n_neurons in architecture_decoder:
            self.__decoder(tf.keras.layers.Dense(n_neurons))

        self.__learning: float = learning

        self.__iterations: int = max_iter

    def generate_with_random_sample(self) -> Image:
        pass

    def __random_sample(self) -> List[int]:
        return tf.random.normal(shape=self.__latent)

    def generate(self, sample: List[int]) -> Image:
        pass

    def fit_dataset(self, x: List[Image]) -> List[float]:
        pass
