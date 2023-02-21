from abc import abstractmethod, ABC
from typing import List

"""
Interface for all the VAE and CVAE implementations. The interface follows the a
sklearn-like structure.
"""


class VAEModel(ABC):

    @abstractmethod
    def fit(self, x: List[List[List[int]]]):  # TODO List[Images]
        pass

    @abstractmethod
    def generate_with_random_sample(self):
        pass

    @abstractmethod
    def generate(self, sample: List[int]):
        pass
