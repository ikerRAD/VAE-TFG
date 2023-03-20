import tensorflow as tf
from typing import Callable, List
import numpy as np


class LossFunctionSelector:
    @staticmethod
    def possible_loss_functions() -> List[str]:
        return ["DKL_MSE"]

    def select(self, function: str) -> Callable:
        if function == "DKL_MSE":
            return self.dkl_mse

    def ___log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    def dkl_mse(
        self,
        z: tf.Tensor,
        means: tf.Tensor,
        logvars: tf.Tensor,
        x: tf.Tensor,
        x_generated: tf.Tensor,
    ) -> float:
        mse: tf.Tensor = tf.keras.losses.mean_squared_error(x, x_generated)
        mse = tf.reduce_sum(mse, axis=[1, 2])
        logpz = self.___log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.___log_normal_pdf(z, means, logvars)
        return tf.reduce_mean(mse - logpz + logqz_x)
