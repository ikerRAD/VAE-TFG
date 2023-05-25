import tensorflow as tf
from typing import Callable, List, Union, Tuple, Dict

#  PRIVATE FUNCTIONS


def __gauss_dkl(mean, logvar, raxis=1):
    return tf.reduce_sum(
        -0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar)), axis=raxis
    )


def __beta_vae_dkl_mse(
    beta: int,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    summary: Dict[str, float] = {}

    mse: tf.Tensor = tf.keras.losses.mean_squared_error(x, x_generated)
    mse = tf.reduce_sum(mse, axis=[1, 2])
    summary["mse"] = tf.reduce_mean(mse)

    dkl = tf.multiply(tf.cast(beta, tf.float32), __gauss_dkl(means, logvars))
    summary["beta-dkl"] = tf.reduce_mean(dkl)

    return tf.reduce_mean(mse + dkl), summary


def __beta_vae_dkl_cross_entropy(
    beta: int,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    summary: Dict[str, float] = {}

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(x, x_generated)
    cross_ent = tf.reduce_sum(cross_ent, axis=[1, 2])
    summary["cross_entropy"] = tf.reduce_mean(cross_ent)

    dkl = tf.multiply(beta, __gauss_dkl(means, logvars))
    summary["beta-dkl"] = tf.reduce_mean(dkl)

    return tf.reduce_mean(cross_ent + dkl), summary


#  COMMON VAE


def dkl_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    summary: Dict[str, float] = {}

    mse: tf.Tensor = tf.keras.losses.mean_squared_error(x, x_generated)
    mse = tf.reduce_sum(mse, axis=[1, 2])
    summary["mse"] = tf.reduce_mean(mse)

    dkl = __gauss_dkl(means, logvars)
    summary["dkl"] = tf.reduce_mean(dkl)

    return tf.reduce_mean(mse + dkl), summary


def dkl_cross_entropy(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    summary: Dict[str, float] = {}

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(x, x_generated)
    cross_ent = tf.reduce_sum(cross_ent, axis=[1, 2])
    summary["cross_entropy"] = tf.reduce_mean(cross_ent)

    dkl = __gauss_dkl(means, logvars)
    summary["dkl"] = tf.reduce_mean(dkl)

    return tf.reduce_mean(cross_ent + dkl), summary


#  BETA VAE


def beta_2_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_mse(2, means, logvars, x, x_generated)


def beta_5_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_mse(5, means, logvars, x, x_generated)


def beta_10_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_mse(10, means, logvars, x, x_generated)


def beta_25_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_mse(25, means, logvars, x, x_generated)


def beta_35_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_mse(35, means, logvars, x, x_generated)


def beta_50_mse(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_mse(50, means, logvars, x, x_generated)


def beta_2_cross_entropy(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_cross_entropy(2, means, logvars, x, x_generated)


def beta_5_cross_entropy(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_cross_entropy(5, means, logvars, x, x_generated)


def beta_10_cross_entropy(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_cross_entropy(10, means, logvars, x, x_generated)


def beta_25_cross_entropy(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_cross_entropy(25, means, logvars, x, x_generated)


def beta_35_cross_entropy(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_cross_entropy(35, means, logvars, x, x_generated)


def beta_50_cross_entropy(
    z: tf.Tensor,
    means: tf.Tensor,
    logvars: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    return __beta_vae_dkl_cross_entropy(50, means, logvars, x, x_generated)


#  SELECTOR


class ImageLossFunctionSelector:
    @staticmethod
    def possible_keys() -> List[str]:
        return [
            "dkl_mse",
            "dkl_cross_entropy",
            "beta_2_mse",
            "beta_5_mse",
            "beta_10_mse",
            "beta_25_mse",
            "beta_35_mse",
            "beta_50_mse",
            "beta_2_cross_entropy",
            "beta_5_cross_entropy",
            "beta_10_cross_entropy",
            "beta_25_cross_entropy",
            "beta_35_cross_entropy",
            "beta_50_cross_entropy",
        ]

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
        if function == "dkl_mse":
            return dkl_mse

        if function == "dkl_cross_entropy":
            return dkl_cross_entropy

        if function == "beta_2_mse":
            return beta_2_mse

        if function == "beta_5_mse":
            return beta_5_mse

        if function == "beta_10_mse":
            return beta_10_mse

        if function == "beta_25_mse":
            return beta_25_mse

        if function == "beta_35_mse":
            return beta_35_mse

        if function == "beta_50_mse":
            return beta_50_mse

        if function == "beta_2_cross_entropy":
            return beta_2_cross_entropy

        if function == "beta_5_cross_entropy":
            return beta_5_cross_entropy

        if function == "beta_10_cross_entropy":
            return beta_10_cross_entropy

        if function == "beta_25_cross_entropy":
            return beta_25_cross_entropy

        if function == "beta_35_cross_entropy":
            return beta_35_cross_entropy

        if function == "beta_50_cross_entropy":
            return beta_50_cross_entropy

        return function
