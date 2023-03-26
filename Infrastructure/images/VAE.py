from typing import List, Optional, Union, Callable
import tensorflow as tf
import numpy as np
from Domain.Exceptions.illegal_architecture_exception import (
    IllegalArchitectureException,
)
from Infrastructure.images.main.image_VAE import ImageVAE
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

"""
Implementation of the most common version of the VAE.
"""


class VAE(ImageVAE):
    def __init__(
        self,
        architecture_encoder: List[int],
        architecture_decoder: List[int],
        encoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        decoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        encoder_output_activation: Union[str, Callable, None] = None,
        decoder_output_activation: Union[str, Callable, None] = None,
        dataset: Optional[List[int]] = None,
        learning_rate: float = 0.0001,
        n_distributions: int = 5,
        max_iter: int = 1000,
        image_length: int = 28,
        image_width: int = 28,
        n_channels: int = 1,
        normalize_data: bool = True,
        discretize_data: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            dataset,
            learning_rate,
            n_distributions,
            max_iter,
            image_length,
            image_width,
            n_channels,
            normalize_data,
            discretize_data,
            *args,
            **kwargs,
        )

        self.__do_checks_for_init(
            architecture_encoder,
            architecture_decoder,
            encoder_activations,
            decoder_activations,
        )

        self._encoder_architecture: List[int] = architecture_encoder
        self._decoder_architecture: List[int] = architecture_decoder
        self._encoder_activations: Optional[
            List[Union[str, Callable, None]]
        ] = encoder_activations
        self._decoder_activations: Optional[
            List[Union[str, Callable, None]]
        ] = decoder_activations
        self._encoder_output_activation: Union[
            str, Callable, None
        ] = encoder_output_activation
        self._decoder_output_activation: Union[
            str, Callable, None
        ] = decoder_output_activation

        n_neurons: int
        for i in range(len(self._encoder_architecture)):
            n_neurons = self._encoder_architecture[i]
            if self._encoder_activations is None:
                self._encoder.add(tf.keras.layers.Dense(n_neurons))
            else:
                self._encoder.add(
                    tf.keras.layers.Dense(
                        n_neurons, activation=self._encoder_activations[i]
                    )
                )
        self._encoder.add(
            tf.keras.layers.Dense(
                self._latent * 2, activation=self._encoder_output_activation
            )
        )

        for i in range(len(self._decoder_architecture)):
            n_neurons = self._decoder_architecture[i]
            if self._decoder_activations is None:
                self._decoder.add(tf.keras.layers.Dense(n_neurons))
            else:
                self._decoder.add(
                    tf.keras.layers.Dense(
                        n_neurons, activation=self._decoder_activations[i]
                    )
                )
        self._decoder.add(
            tf.keras.layers.Dense(
                self._length * self._width * self._channels,
                activation=self._decoder_output_activation,
            )
        )

    def get_decoder_architecture(self) -> List[int]:
        return self._decoder_architecture

    def get_encoder_architecture(self) -> List[int]:
        return self._encoder_architecture

    def get_decoder_activations(
        self,
    ) -> Optional[List[Union[str, Callable, None]]]:
        return self._decoder_activations

    def get_encoder_activations(
        self,
    ) -> Optional[List[Union[str, Callable, None]]]:
        return self._encoder_activations

    def get_decoder_output_activations(self) -> Union[str, Callable, None]:
        return self._decoder_output_activation

    def get_encoder_output_activations(self) -> Union[str, Callable, None]:
        return self._encoder_output_activation

    def __do_checks_for_init(
        self,
        architecture_encoder: List[int],
        architecture_decoder: List[int],
        encoder_activations: Optional[List[Union[str, Callable, None]]],
        decoder_activations: Optional[List[Union[str, Callable, None]]],
    ) -> None:
        if np.any(np.array(architecture_encoder) <= 0):
            raise IllegalArchitectureException(
                "The architecture of the encoder cannot have layers with values lower than 1"
            )

        if np.any(np.array(architecture_decoder) <= 0):
            raise IllegalArchitectureException(
                "The architecture of the decoder cannot have layers with values lower than 1"
            )

        if len(architecture_encoder) == 0:
            raise IllegalArchitectureException(
                "The architecture of the encoder cannot be empty"
            )

        if len(architecture_decoder) == 0:
            raise IllegalArchitectureException(
                "The architecture of the decoder cannot be empty"
            )

        if architecture_encoder[-1] < self._latent * 2:
            raise IllegalArchitectureException(
                f"The last layer of the encoder cannot be lower than {self._latent * 2}"
            )

        if architecture_decoder[0] < self._latent:
            raise IllegalArchitectureException(
                f"The first layer of the decoder cannot be lower than {self._latent}"
            )

        if encoder_activations is not None and (
            len(encoder_activations) != len(architecture_encoder)
        ):
            raise IllegalArchitectureException(
                f"The number of encoder activation functions must be the same as the number of encoding layers"
            )

        if decoder_activations is not None and (
            len(decoder_activations) != len(architecture_decoder)
        ):
            raise IllegalArchitectureException(
                f"The number of decoder activation functions must be the same as the number of decoding layers"
            )
