import tensorflow as tf
from typing import Callable, List, Union, Tuple, Dict
import numpy as np


def __log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis,
    )


def dkl_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    mse: tf.Tensor = tf.keras.losses.mean_squared_error(x, x_generated)
    mse = tf.reduce_sum(mse, axis=[1, 2])
    logpz = __log_normal_pdf(z, 0.0, 0.0)
    logqz_x = __log_normal_pdf(z, means, logvars)
    return tf.reduce_mean(mse - logpz + logqz_x), {}


class ImageLossFunctionSelector:
    @staticmethod
    def possible_keys() -> List[str]:
        return ["dkl_mse"]

    @staticmethod
    def select(
        function: Union[
            str,
            Callable[
                [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
                Tuple[float, Dict[str, float]],
            ],
        ]
    ) -> Callable[
        [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        Tuple[float, Dict[str, float]],
    ]:
        if function == "DKL_MSE":
            return dkl_mse
        else:
            return function
