import tensorflow as tf
from typing import Callable, List
import numpy as np
from Domain.VAEModel import VAEModel


class LossFunctionSelector:
    def possible_loss_functions(self) -> List[str]:
        return ["DKL_MSE"]

    def select(self, function: str) -> Callable:
        if function == "DKL_MSE":
            return self.dkl_mse

    def dkl_mse(self, model: VAEModel, x) -> float:
        pass
