from typing import List, Optional, Union, Callable, Tuple, Dict
import tensorflow as tf
import numpy as np

from src.project.Infrastructure.images.main.image_VAE import ImageVAE
from src.project.domain.Exceptions.illegal_architecture_exception import (
    IllegalArchitectureException,
)
from src.project.domain.Exceptions.illegal_value_exception import IllegalValueException
from src.utils.losses.images.application.image_loss_function_selector import (
    ImageLossFunctionSelector,
)

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
        decoder_output_size: int = 3,
        dataset: Optional[List] = None,
        loss: Union[
            str,
            Callable[
                [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
                Tuple[float, Dict[str, float]],
            ],
        ] = ImageLossFunctionSelector.possible_keys()[0],
        learning_rate: float = 0.00001,
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
        super().__init__(
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
            decoder_output_size,
        )

        self._encoder_architecture: List[Union[int, str]] = encoder_architecture
        self._decoder_architecture: List[int] = decoder_architecture
        self._encoder_sizes: List[int] = encoder_sizes
        self._decoder_sizes: List[int] = decoder_sizes
        self._encoder_strides: List[int] = encoder_strides
        self._decoder_strides: List[int] = decoder_strides

        self._decoder_input_reshape: Tuple[int, int, int] = decoder_input_reshape

        self._encoder_activations: Optional[
            List[Union[str, Callable, None]]
        ] = encoder_activations
        self._decoder_activations: Optional[
            List[Union[str, Callable, None]]
        ] = decoder_activations
        self._encoder_output_activation: Union[
            str, Callable, None
        ] = encoder_output_activation
        self._decoder_input_activation: Union[
            str, Callable, None
        ] = decoder_input_activation
        self._decoder_output_activation: Union[
            str, Callable, None
        ] = decoder_output_activation
        self._decoder_output_size: int = decoder_output_size

        self._encoder.add(
            tf.keras.layers.InputLayer(
                input_shape=(self._height, self._width, self._channels)
            )
        )
        filter_or_pooling: Union[int, str]
        sizes: int
        strides: int
        for i in range(len(self._encoder_architecture)):
            filter_or_pooling = self._encoder_architecture[i]
            sizes = self._encoder_sizes[i]
            strides = self._encoder_strides[i]
            if type(filter_or_pooling) is str:
                is_max_pool: bool = filter_or_pooling != "max_pool"
                is_average_pool: bool = filter_or_pooling != "average_pool"
                if is_max_pool:
                    self._encoder.add(
                        tf.keras.layers.MaxPooling2D(
                            pool_size=sizes,
                            strides=strides,
                        )
                    )
                elif is_average_pool:
                    self._encoder.add(
                        tf.keras.layers.AveragePooling2D(
                            pool_size=sizes,
                            strides=strides,
                        )
                    )
            elif type(filter_or_pooling) is int:
                if self._encoder_activations is None:
                    self._encoder.add(
                        tf.keras.layers.Conv2D(
                            filters=filter_or_pooling,
                            kernel_size=sizes,
                            strides=strides,
                        )
                    )
                else:
                    self._encoder.add(
                        tf.keras.layers.Conv2D(
                            filters=filter_or_pooling,
                            kernel_size=sizes,
                            strides=strides,
                            activation=self._encoder_activations[i],
                        )
                    )
        self._encoder.add(tf.keras.layers.Flatten())
        self._encoder.add(
            tf.keras.layers.Dense(
                self._latent * 2, activation=self._encoder_output_activation
            )
        )

        self._decoder.add(tf.keras.layers.InputLayer(input_shape=(self._latent,)))
        self._decoder.add(
            tf.keras.layers.Dense(
                self._decoder_input_reshape[0]
                * self._decoder_input_reshape[1]
                * self._decoder_input_reshape[2],
                activation=self._decoder_input_activation,
            )
        )
        self._decoder.add(
            tf.keras.layers.Reshape(target_shape=self._decoder_input_reshape)
        )
        filters: int
        for i in range(len(self._decoder_architecture)):
            filters = self._decoder_architecture[i]
            sizes = self._decoder_sizes[i]
            strides = self._decoder_strides[i]
            if self._decoder_activations is None:
                self._decoder.add(
                    tf.keras.layers.Conv2DTranspose(
                        filters=filters,
                        kernel_size=sizes,
                        strides=strides,
                        padding="same",
                    )
                )
            else:
                self._decoder.add(
                    tf.keras.layers.Conv2DTranspose(
                        filters=filters,
                        kernel_size=sizes,
                        strides=strides,
                        activation=self._encoder_activations[i],
                        padding="same",
                    )
                )
        self._decoder.add(
            tf.keras.layers.Conv2DTranspose(
                filters=self._channels,
                kernel_size=self._decoder_output_size,
                strides=1,
                padding="same",
            )
        )

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
        decoder_output_size: int,
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

        input_height: int
        input_width: int
        input_height, input_width, _ = decoder_input_reshape
        for stride in decoder_strides:
            input_height *= stride
            input_width *= stride

        incorrect_height: bool = input_height != self._height
        if incorrect_height:
            raise IllegalArchitectureException(
                f"The resulting height of the decoder's output images would be {input_height}"
                f" and should be {self._height}. Change either the values of the decoder's strides"
                " or the input shape"
            )

        incorrect_width: bool = input_width != self._width
        if incorrect_width:
            raise IllegalArchitectureException(
                f"The resulting width of the decoder's output images would be {input_width}"
                f" and should be {self._width}. Change either the values of the decoder's strides"
                " or the input shape"
            )

        if decoder_output_size <= 0:
            raise IllegalValueException(
                "The 'decoder output size' parameter cannot be less than or equal to zero."
            )

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
