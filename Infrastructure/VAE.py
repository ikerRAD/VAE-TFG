from typing import List
import tensorflow as tf
import numpy as np
from Exceptions.IllegalArchitectureException import IllegalArchitectureException
from Domain.VAEModel import VAEModel

"""
Implementation of the most common version of the VAE.
"""


class VAE(VAEModel):

    def __init__(self,
                 architecture_encoder: List[int],
                 architecture_decoder: List[int],
                 learning_rate: float = 0.001,
                 n_distributions: int = 5,
                 max_iter: int = -1):

        if len(architecture_encoder) == 0:
            raise IllegalArchitectureException(f"The architecture of the encoder cannot be empty")

        if len(architecture_decoder) == 0:
            raise IllegalArchitectureException(f"The architecture of the decoder cannot be empty")

        if architecture_encoder[-1] < n_distributions * 2:
            raise IllegalArchitectureException(
                f"The last layer of the encoder cannot be lower than {n_distributions * 2}")

        if architecture_decoder[0] < n_distributions * 2:
            raise IllegalArchitectureException(
                f"The first layer of the decoder cannot be lower than {n_distributions * 2}")

        self.encoder = architecture_encoder
        self.latent = n_distributions * 2
        self.decoder = architecture_decoder

        self.learning_rate = learning_rate

        self.iterations = max_iter
