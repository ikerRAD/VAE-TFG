import tensorflow as tf
from typing import Callable, List
import numpy as np
from Domain.VAEModel import VAEModel
from Utils.proyect_typings import Image


class LossFunctionSelector:
    def possible_loss_functions(self) -> List[str]:
        return ["DKL_MSE"]

    def select(self, function: str) -> Callable:
        if function == "DKL_MSE":
            return self.dkl_mse

    def dkl_mse(self, model: VAEModel, x: Image) -> float:
        encoded: (List[float], List[float]) = model.encode(x)
        means: List[float] = encoded[0]
        logvars: List[float] = encoded[1]
        z: List[float] = model.reparameterize(means, logvars)
        x_out: Image = model.decode(z)

        mse: List[float] = tf.keras.losses.mean_squared_error(x, x_out)
        loss_mse: float = tf.reduce_sum(mse) * 100

        dkl: List[float] = tf.reduce_sum(
            -0.5 * ((z - 0.0) ** 2.0 * tf.exp(-0.0) + 0.0 + tf.math.log(2.0 * np.pi)),
            axis=1,
        )
        loss_dkl: float = tf.reduce_mean(dkl)

        return loss_mse - loss_dkl
