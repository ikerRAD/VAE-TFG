from typing import List, Optional, Union, Callable, Tuple
import tensorflow as tf
import numpy as np

from project.domain.Exceptions.illegal_architecture_exception import (
    IllegalArchitectureException,
)
from infrastructure.images.main.image_VAE import ImageVAE


"""
Implementation of the most common version of the CVAE.
"""


class CVAE(ImageVAE):
    def __init__(
        self,
        encoder_architecture: List[Union[int, str]],
        decoder_architecture: List[int],
        encoder_sizes: List[int],
        decoder_sizes: List[int],
        encoder_strides: List[int],
        decoder_strides: List[int],
        decoder_input_reshape: Tuple[int, int, int],
        encoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        decoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        encoder_output_activation: Union[str, Callable, None] = None,
        decoder_input_activation: Union[str, Callable, None] = None,
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
            encoder_architecture,
            decoder_architecture,
            encoder_sizes,
            decoder_sizes,
            encoder_strides,
            decoder_strides,
            decoder_input_reshape,
            encoder_activations,
            decoder_activations,
        )

        """ REVISAR """
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
        """ REVISAR """

    def __do_checks_for_init(
        self,
        encoder_architecture: List[Union[int, str]],
        decoder_architecture: List[int],
        encoder_sizes: List[int],
        decoder_sizes: List[int],
        encoder_strides: List[int],
        decoder_strides: List[int],
        decoder_input_reshape: Tuple[int, int, int],
        encoder_activations: Optional[List[Union[str, Callable, None]]],
        decoder_activations: Optional[List[Union[str, Callable, None]]],
    ) -> None:
        for encoder_unit in encoder_architecture:
            if type(encoder_unit) is str:
                is_not_max_pool: bool = encoder_unit != "max_pool"
                is_not_average_pool: bool = encoder_unit != "average_pool"
                if is_not_max_pool and is_not_average_pool:
                    raise IllegalArchitectureException(
                        f"The architecture of the encoder cannot understand {encoder_unit}."
                        f" It should be either 'max_pool', 'average_pool' or any integer value greater than 0"
                    )
            elif type(encoder_unit) is int:
                if encoder_unit <= 0:
                    raise IllegalArchitectureException(
                        "The architecture of the encoder cannot have filters with a size lower than 1"
                    )

        if np.any(np.array(decoder_architecture) <= 0):
            raise IllegalArchitectureException(
                "The architecture of the decoder cannot have layers with values lower than 1"
            )

        if len(encoder_architecture) == 0:
            raise IllegalArchitectureException(
                "The architecture of the encoder cannot be empty"
            )

        if len(decoder_architecture) == 0:
            raise IllegalArchitectureException(
                "The architecture of the decoder cannot be empty"
            )

        if np.any(np.array(encoder_sizes) <= 0):
            raise IllegalArchitectureException(
                "The sizes of the encoder cannot have values lower than 1"
            )

        if np.any(np.array(decoder_sizes) <= 0):
            raise IllegalArchitectureException(
                "The sizes of the decoder cannot have values lower than 1"
            )
        if np.any(np.array(encoder_strides) <= 0):
            raise IllegalArchitectureException(
                "The strides of the encoder cannot have values lower than 1"
            )

        if np.any(np.array(decoder_strides) <= 0):
            raise IllegalArchitectureException(
                "The strides of the decoder cannot have values lower than 1"
            )

        """Ver que la entrada al decoder es decente"""

        if encoder_activations is not None and (
            len(encoder_activations) != len(encoder_architecture)
        ):
            raise IllegalArchitectureException(
                f"The number of encoder activation functions must be the same as the number of encoding layers"
            )

        if decoder_activations is not None and (
            len(decoder_activations) != len(decoder_architecture)
        ):
            raise IllegalArchitectureException(
                f"The number of decoder activation functions must be the same as the number of decoding layers"
            )

        if len(encoder_sizes) != len(encoder_architecture):
            raise IllegalArchitectureException(
                f"The number of encoder sizes must be the same as the number of encoding layers"
            )

        if len(decoder_sizes) != len(decoder_architecture):
            raise IllegalArchitectureException(
                f"The number of decoder sizes must be the same as the number of decoding layers"
            )

        if len(encoder_strides) != len(encoder_architecture):
            raise IllegalArchitectureException(
                f"The number of encoder strides must be the same as the number of encoding layers"
            )

        if len(decoder_strides) != len(decoder_architecture):
            raise IllegalArchitectureException(
                f"The number of decoder strides must be the same as the number of decoding layers"
            )
