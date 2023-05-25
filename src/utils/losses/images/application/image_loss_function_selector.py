import tensorflow as tf
from typing import Callable, List, Union, Tuple, Dict, Optional


#  PRIVATE FUNCTIONS
from numpy import nan


def __beta_fn(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    if a == nan:
        a = 10000.
    return tf.exp(tf.math.lgamma(a) + tf.math.lgamma(b) - tf.math.lgamma(a + b))


def __sb_dkl(
    alpha: tf.Tensor,
    beta: tf.Tensor,
    prior_alpha: float = 1.0,
    prior_beta: float = 5.0,
    raxis: Optional[int] = 1,
) -> tf.Tensor:
    kl: tf.Tensor = 1.0 / (1.0 + alpha * beta) * __beta_fn(1.0 / alpha, beta)
    kl = kl + 1.0 / (2.0 + alpha * beta) * __beta_fn(2.0 / alpha, beta)
    kl = kl + 1.0 / (3.0 + alpha * beta) * __beta_fn(3.0 / alpha, beta)
    kl = kl + 1.0 / (4.0 + alpha * beta) * __beta_fn(4.0 / alpha, beta)
    kl = kl + 1.0 / (5.0 + alpha * beta) * __beta_fn(5.0 / alpha, beta)
    kl = kl + 1.0 / (6.0 + alpha * beta) * __beta_fn(6.0 / alpha, beta)
    kl = kl + 1.0 / (7.0 + alpha * beta) * __beta_fn(7.0 / alpha, beta)
    kl = kl + 1.0 / (8.0 + alpha * beta) * __beta_fn(8.0 / alpha, beta)
    kl = kl + 1.0 / (9.0 + alpha * beta) * __beta_fn(9.0 / alpha, beta)
    kl = kl + 1.0 / (10.0 + alpha * beta) * __beta_fn(10.0 / alpha, beta)
    kl = kl * (prior_beta - 1.0) * beta

    psi_b_taylor_approx: tf.Tensor = (
        tf.math.log(beta) - 1.0 / 2.0 * beta - 1.0 / 12.0 * beta**2.0
    )

    kl = kl + (alpha - prior_alpha) / alpha * (
        -0.57721 - psi_b_taylor_approx - 1.0 / beta
    )

    kl = (
        kl
        + tf.math.log(alpha * beta)
        + tf.math.log(
            __beta_fn(
                tf.convert_to_tensor(prior_alpha), tf.convert_to_tensor(prior_beta)
            )
        )
    )

    kl = kl + -(beta - 1.0) / beta
    return tf.reduce_sum(kl, axis=raxis)


def __gauss_dkl(
    mean: tf.Tensor, logvar: tf.Tensor, raxis: Optional[int] = 1
) -> tf.Tensor:
    return tf.reduce_sum(
        -0.5 * (tf.add(1.0, logvar) - tf.square(mean) - tf.exp(logvar)), axis=raxis
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


#  STICK-BREAKING VAE

@tf.function
def sb_mse(
    z: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    summary: Dict[str, float] = {}

    mse: tf.Tensor = tf.keras.losses.mean_squared_error(x, x_generated)
    mse = tf.reduce_sum(mse, axis=[1, 2])
    summary["mse"] = tf.reduce_mean(mse)

    sb_dkl = __sb_dkl(alpha, beta)
    print(sb_dkl)
    summary["sb_dkl"] = tf.reduce_mean(sb_dkl)

    return tf.reduce_mean(mse + sb_dkl), summary


def sb_cross_entropy(
    z: tf.Tensor,
    alpha: tf.Tensor,
    beta: tf.Tensor,
    x: tf.Tensor,
    x_generated: tf.Tensor,
) -> Tuple[float, Dict[str, float]]:
    summary: Dict[str, float] = {}

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(x, x_generated)
    cross_ent = tf.reduce_sum(cross_ent, axis=[1, 2])
    summary["cross_entropy"] = tf.reduce_mean(cross_ent)

    sb_dkl = __sb_dkl(alpha, beta)
    summary["sb_dkl"] = tf.reduce_mean(sb_dkl)

    return tf.reduce_mean(cross_ent + sb_dkl), summary


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
            "sb_mse",
            "sb_cross_entropy",
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

        if function == "sb_mse":
            return sb_mse

        if function == "sb_cross_entropy":
            return sb_cross_entropy

        return function
