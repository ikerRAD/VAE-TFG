from typing import List, Optional, Tuple, Union, Callable
import tensorflow as tf
import numpy as np

from Domain.Exceptions.illegal_architecture_exception import (
    IllegalArchitectureException,
)
from Domain.Exceptions.no_more_batches_exception import NoMoreBatchesException
from Domain.VAEModel import VAEModel
from Domain.Exceptions.illegal_value_exception import IllegalValueException
from Utils.batch_calculators import (
    Batch,
    CommonBatch,
    StrictBatch,
    CyclicBatch,
    RandomBatch,
    RandomStrictBatch,
)
from tensorflow.python.ops.numpy_ops import np_config

from Utils.loss_function_selector import LossFunctionSelector

np_config.enable_numpy_behavior()

"""
Implementation of the most common version of the VAE.
"""


class VAE(VAEModel):
    def __init__(
        self,
        architecture_encoder: List[int],
        architecture_decoder: List[int],
        encoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        decoder_activations: Optional[List[Union[str, Callable, None]]] = None,
        encoder_output_activation: Union[str, Callable, None] = None,
        decoder_output_activation: Union[str, Callable, None] = None,
        dataset: Optional[List[int]] = None,
        learning: float = 0.0001,
        n_distributions: int = 5,
        max_iter: int = 1000,
        image_length: int = 28,
        image_width: int = 28,
        n_channels: int = 1,
        normalize_pixels: bool = True,
        discretize_pixels: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.__do_checks_for_init(
            architecture_encoder,
            architecture_decoder,
            encoder_activations,
            decoder_activations,
            learning,
            n_distributions,
            max_iter,
        )

        self.__do_checks_for_image_shape(
            image_length,
            image_width,
            n_channels,
        )

        self.__epsilon: Optional[tf.Tensor] = None
        self.__length: int
        self.__width: int
        self.__channels: int
        self.__train_images: tf.Tensor

        if dataset is None:
            (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
            self.__length = 28
            self.__width = 28
            self.__channels = 1
        else:
            train_images = tf.convert_to_tensor(dataset, dtype=tf.float32)
            self.__length = image_length
            self.__width = image_width
            self.__channels = n_channels

        self.__train_images = train_images.reshape(
            (train_images.shape[0], self.__length, self.__width, self.__channels)
        )

        normalizer: float = 0.0
        if normalize_pixels:
            normalizer = 255.0

        self.__train_images = self.__train_images / normalizer

        if discretize_pixels:
            self.__train_images = tf.cast(
                tf.where(self.__train_images > 0.5, 1.0, 0.0), tf.float32
            )

        self.__encoder_architecture: List[int] = architecture_encoder
        self.__decoder_architecture: List[int] = architecture_decoder
        self.__encoder_activations: Optional[
            List[Union[str, Callable, None]]
        ] = encoder_activations
        self.__decoder_activations: Optional[
            List[Union[str, Callable, None]]
        ] = decoder_activations
        self.__encoder_output_activation: Union[
            str, Callable, None
        ] = encoder_output_activation
        self.__decoder_output_activation: Union[
            str, Callable, None
        ] = decoder_output_activation
        self.__latent: int = n_distributions

        self.__learning: float = learning
        self.__optimizer = tf.keras.optimizers.Adam(self.__learning)
        self.__iterations: int = max_iter

        n_neurons: int
        self.__encoder = tf.keras.Sequential()
        for i in range(len(self.__encoder_architecture)):
            n_neurons = self.__encoder_architecture[i]
            if self.__encoder_activations is None:
                self.__encoder.add(tf.keras.layers.Dense(n_neurons))
            else:
                self.__encoder.add(
                    tf.keras.layers.Dense(
                        n_neurons, activation=self.__encoder_activations[i]
                    )
                )
        self.__encoder.add(
            tf.keras.layers.Dense(
                self.__latent * 2, activation=self.__encoder_output_activation
            )
        )

        self.__decoder = tf.keras.Sequential()
        for i in range(len(self.__decoder_architecture)):
            n_neurons = self.__decoder_architecture[i]
            if self.__decoder_activations is None:
                self.__decoder.add(tf.keras.layers.Dense(n_neurons))
            else:
                self.__decoder.add(
                    tf.keras.layers.Dense(
                        n_neurons, activation=self.__decoder_activations[i]
                    )
                )
        self.__decoder.add(
            tf.keras.layers.Dense(
                self.__length * self.__width * self.__channels,
                activation=self.__decoder_output_activation,
            )
        )

        loss_selector = LossFunctionSelector()
        self.__loss_function = loss_selector.select("DKL_MSE")

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def __train_step(self, x: tf.Tensor) -> float:
        with tf.GradientTape() as tape:
            means: tf.Tensor
            logvars: tf.Tensor
            means, logvars = self.__encode(x)
            z: tf.Tensor = self.__reparameterize(means, logvars)
            x_generated: tf.Tensor = self.__decode(z)

            loss: float = self.__loss_function(z, means, logvars, x, x_generated)

            gradients = tape.gradient(loss, self.trainable_variables)
            self.__optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return loss

    def __iterate_without_batch(
        self, train_images: tf.Tensor, generate_images
    ) -> List[float]:
        loss_values: List[float] = []

        for iteration in range(1, self.__iterations + 1):
            print(f"Iteration number {iteration}")
            if generate_images:
                self.generate_random_images(save=False)

            partial_loss: List[float] = [self.__train_step(train_images)]

            loss_values.append(tf.reduce_mean(partial_loss))

        return loss_values

    def __iterate_with_batch(self, the_batch: Batch, generate_images) -> List[float]:
        loss_values: List[float] = []
        train_images: tf.Tensor

        try:
            for iteration in range(1, self.__iterations + 1):
                print(f"Iteration number {iteration}")
                if generate_images:
                    self.generate_random_images(save=False)

                train_images = the_batch.next()
                partial_loss: List[float] = [self.__train_step(train_images)]

                loss_values.append(tf.reduce_mean(partial_loss))
        except NoMoreBatchesException as termination:
            print(str(termination))

        return loss_values

    def fit_dataset(
        self,
        return_loss: bool = False,
        batch_size: int = 100,
        batch_type: Optional[Union[str, Batch]] = None,
        generate_images: bool = True,
    ) -> Optional[List[float]]:
        loss_values: List[float]
        if batch_type is None:
            self.__epsilon = tf.random.normal(
                shape=(self.__train_images.shape[0], self.__latent)
            )
            loss_values = self.__iterate_without_batch(
                self.__train_images, generate_images
            )
        else:
            self.__epsilon = tf.random.normal(shape=(batch_size, self.__latent))
            the_batch: Batch
            if batch_type == "common":
                the_batch = CommonBatch()
            elif batch_type == "strict":
                the_batch = StrictBatch()
            elif batch_type == "cyclic":
                the_batch = CyclicBatch()
            elif batch_type == "random":
                the_batch = RandomBatch()
            elif batch_type == "random_strict":
                the_batch = RandomStrictBatch()
            else:
                the_batch = batch_type

            the_batch.set_up(self.__train_images, batch_size)
            loss_values = self.__iterate_with_batch(the_batch, generate_images)

        if return_loss:
            return loss_values

    def __random_sample(self, n_samples: int = 1) -> tf.Tensor:
        return tf.random.normal(shape=(n_samples, self.__latent))

    def generate_with_random_sample(self, n_samples: int = 1) -> tf.Tensor:
        sample: tf.Tensor = self.__random_sample(n_samples)
        return self.__decode(sample)

    def generate_with_one_sample(
        self, sample: List[float], n_samples: int = 1
    ) -> tf.Tensor:
        samples: tf.Tensor = tf.convert_to_tensor(
            [sample for _ in range(n_samples)], tf.float32
        )
        return self.__decode(samples)

    def generate_with_multiple_samples(self, samples: tf.Tensor) -> tf.Tensor:
        return self.__decode(samples)

    def __reparameterize(self, means: tf.Tensor, logvars: tf.Tensor) -> tf.Tensor:
        return self.__epsilon * tf.exp(logvars * 0.5) + means
        # TODO does the e^(logvars*0.5) make sense?

    def __encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x_resized: tf.Tensor = x.reshape(
            (x.shape[0], self.__length * self.__width * self.__channels)
        )
        means, logvars = tf.split(
            self.__encoder(x_resized), num_or_size_splits=2, axis=1
        )
        return means, logvars

    def __decode(self, z: tf.Tensor) -> tf.Tensor:
        images_resized: tf.Tensor = self.__decoder(z).reshape(
            (z.shape[0], self.__length, self.__width, self.__channels)
        )
        return images_resized

    def encode_and_decode(self, x: tf.Tensor) -> tf.Tensor:
        means: tf.Tensor
        logvars: tf.Tensor

        means, logvars = self.__encode(x)
        z: tf.Tensor = self.__reparameterize(means, logvars)
        x_generated: tf.Tensor = self.__decode(z)

        return x_generated

    def __do_checks_for_init(
        self,
        architecture_encoder: List[int],
        architecture_decoder: List[int],
        encoder_activations: Optional[List[Union[str, Callable, None]]],
        decoder_activations: Optional[List[Union[str, Callable, None]]],
        learning: float,
        n_distributions: int,
        max_iter: int,
    ) -> None:
        if learning <= 0:
            raise IllegalValueException(
                "The 'learning rate' or 'learning_step' parameter cannot be less than or equal to zero."
            )

        if max_iter <= 0:
            raise IllegalValueException(
                "The maximum number of iterations cannot be less than or equal to zero."
            )

        if n_distributions <= 0:
            raise IllegalArchitectureException(
                "The number of distributions cannot be less than or equal to zero."
            )

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

        if architecture_encoder[-1] < n_distributions * 2:
            raise IllegalArchitectureException(
                f"The last layer of the encoder cannot be lower than {n_distributions * 2}"
            )

        if architecture_decoder[0] < n_distributions:
            raise IllegalArchitectureException(
                f"The first layer of the decoder cannot be lower than {n_distributions}"
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

    def __do_checks_for_image_shape(
        self,
        image_length,
        image_width,
        n_channels,
    ) -> None:
        if image_length <= 0:
            raise IllegalValueException(
                "The 'image length' parameter cannot be less than or equal to zero."
            )

        if image_width <= 0:
            raise IllegalValueException(
                "The 'image width' parameter cannot be less than or equal to zero."
            )

        if n_channels <= 0:
            raise IllegalValueException(
                "The 'number of channels' parameter cannot be less than or equal to zero. "
                + "It usually is 1 if there is just one channel, 3 if the image follows the "
                + "standarized RGB structure or 4 if a channel is added for opacity"
            )
