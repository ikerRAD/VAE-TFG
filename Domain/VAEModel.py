from abc import abstractmethod
from typing import List

"""
Interface for all the VAE and CVAE implementations. The interface follows the a
sklearn-like structure.
"""


class VAEModel:

    @abstractmethod
    def fit(self, x: List[List[int]]):
        pass

    @abstractmethod
    def generate_with_random_sample(self):
        pass

    @abstractmethod
    def generate(self, sample: List[int]):
        pass
