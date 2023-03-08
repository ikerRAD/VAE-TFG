from typing import List
from Domain.VAEModel import VAEModel
import tensorflow as tf
import numpy as np

"""
Implementation of the most common version of the CVAE.
"""


class CVAE(VAEModel):

    def __init__(self,
                 # TODO hyperparams
                 learning_rate: float = 0.001,
                 n_distributions: int = 5,
                 max_iter: int = -1) -> None:
        self.latent = n_distributions * 2

        self.learning_rate = learning_rate

        self.iterations = max_iter
