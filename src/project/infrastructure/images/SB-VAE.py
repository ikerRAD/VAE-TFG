from typing import List, Optional, Union, Callable, Dict, Tuple
import tensorflow as tf

from src.project.infrastructure.images.VAE import VAE
from src.project.domain.exceptions.illegal_architecture_exception import (
    IllegalArchitectureException,
)
from src.utils.losses.images.application.image_loss_function_selector import (
    ImageLossFunctionSelector,
)


ALLOWED_LOSSES = [
    ImageLossFunctionSelector.ImageLosses.STICKBREAKING_MSE.value,
    ImageLossFunctionSelector.ImageLosses.STICKBREAKING_CROSS_ENTROPY.value,
    ImageLossFunctionSelector.ImageLosses.SB_BETA2_MSE,
    ImageLossFunctionSelector.ImageLosses.SB_BETA5_MSE,
    ImageLossFunctionSelector.ImageLosses.SB_BETA05_MSE,
    ImageLossFunctionSelector.ImageLosses.SB_BETA02_MSE,
    ImageLossFunctionSelector.ImageLosses.SB_BETA2_CROSS_ENTROPY,
    ImageLossFunctionSelector.ImageLosses.SB_BETA5_CROSS_ENTROPY,
    ImageLossFunctionSelector.ImageLosses.SB_BETA05_CROSS_ENTROPY,
    ImageLossFunctionSelector.ImageLosses.SB_BETA02_CROSS_ENTROPY,
]

SOFTPLUS_ROOF = 1.0

"""
Implementation of the most common version of the VAE.
"""


class SBVAE(VAE):
    def __init__(
        self,
        encoder_architecture: List[int],
        decoder_architecture: List[int],
        encoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        decoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        decoder_output_activation: Union[str, Callable, None] = None,
        dataset: Optional[List] = None,
        loss: Union[
            str,
            Callable[
                [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
                Tuple[float, Dict[str, float]],
            ],
        ] = ImageLossFunctionSelector.ImageLosses.STICKBREAKING_MSE.value,
        learning_rate: float = 0.0001,
        n_distributions: int = 5,
        max_iter: int = 1000,
        image_height: int = 28,
        image_width: int = 28,
        n_channels: int = 1,
        normalize_data: bool = True,
        discretize_data: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self.__do_loss_check(
            loss,
        )

        super().__init__(
            encoder_architecture,
            decoder_architecture,
            encoder_activations,
            decoder_activations,
            lambda x, name=None: tf.maximum(tf.math.softplus(x, name), SOFTPLUS_ROOF),
            decoder_output_activation,
            dataset,
            loss,
            learning_rate,
            n_distributions,
            max_iter,
            image_height,
            image_width,
            n_channels,
            normalize_data,
            discretize_data,
            *args,
            **kwargs,
        )

    def __do_loss_check(
        self,
        loss: Union[
            str,
            Callable[
                [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
                Tuple[float, Dict[str, float]],
            ],
        ],
    ) -> None:
        if loss in ImageLossFunctionSelector.possible_keys():
            if loss not in ALLOWED_LOSSES:
                raise IllegalArchitectureException(
                    f"'{loss}' is not a valid loss function for this model"
                )
