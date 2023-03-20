from typing import List, Optional, Tuple, Union
import tensorflow as tf
import numpy as np
from numpy import ndarray

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

np_config.enable_numpy_behavior()

"""
Implementation of the most common version of the VAE.
"""


class VAE(VAEModel):
    def __init__(
        self,
        architecture_encoder: List[int],
        architecture_decoder: List[int],
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
            learning,
            n_distributions,
            max_iter,
        )

        self.__do_checks_for_image_shape(
            image_length,
            image_width,
            n_channels,
        )

        self.__epsilon: Optional[ndarray] = None
        self.__length:int
        self.__width:int
        self.__channels:int
        self.__train_images:ndarray

        if dataset is None:
            (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
            self.__length = 28
            self.__width = 28
            self.__channels = 1
        else:
            train_images = np.array(dataset)
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
            self.__train_images = np.where(self.__train_images > 0.5, 1.0, 0.0).astype(
                "float32"
            )

        self.__encoder_architecture: List[int] = architecture_encoder
        self.__decoder_architecture: List[int] = architecture_decoder
        self.__latent: int = n_distributions

        self.__learning: float = learning
        self.__optimizer = tf.keras.optimizers.Adam(self.__learning)
        self.__iterations: int = max_iter

        self.__encoder = tf.keras.Sequential()
        for n_neurons in self.__encoder_architecture:
            self.__encoder.add(tf.keras.layers.Dense(n_neurons))
        self.__encoder.add(tf.keras.layers.Dense(self.__latent * 2))

        self.__decoder = tf.keras.Sequential()
        for n_neurons in self.__decoder_architecture:
            self.__decoder.add(tf.keras.layers.Dense(n_neurons))
        self.__decoder.add(
            tf.keras.layers.Dense(self.__length * self.__width * self.__channels)
        )

        # loss_selector = LossFunctionSelector() TODO recuperar
        # self.__loss_function = loss_selector.select('DKL_MSE')

    def __train_step(self, x: ndarray) -> float:
        with tf.GradientTape() as tape:
            # loss: float = self.__loss_function(self, x)
            loss = self.___loss(x)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.__optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return loss

    def __iterate_without_batch(
        self, train_images: ndarray, generate_images
    ) -> List[float]:
        loss_values: List[float] = []

        for iteration in range(1, self.__iterations + 1):
            print(f"Iteration number {iteration}")
            if generate_images:
                self.generate_and_save_random_images()

            partial_loss: List[float] = [self.__train_step(train_images)]

            loss_values.append(tf.reduce_mean(partial_loss))

        return loss_values

    def __iterate_with_batch(self, the_batch: Batch, generate_images) -> List[float]:
        loss_values: List[float] = []
        train_images: ndarray

        try:
            for iteration in range(1, self.__iterations + 1):
                print(f"Iteration number {iteration}")
                if generate_images:
                    self.generate_and_save_random_images()

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

    def __random_sample(self, n_samples: int = 1) -> ndarray:
        return np.array(tf.random.normal(shape=(n_samples, self.__latent)))

    def generate_with_random_sample(self, n_samples: int = 1) -> ndarray:
        sample: ndarray = self.__random_sample(n_samples)
        return np.array(self.__decode(sample))

    def generate_with_one_sample(
        self, sample: List[float], n_samples: int = 1
    ) -> ndarray:
        samples: ndarray = np.array([sample for _ in range(n_samples)])
        return np.array(self.__decode(samples))

    def generate_with_multiple_samples(self, samples: ndarray) -> ndarray:
        return np.array(self.__decode(samples))

    def __reparameterize(self, means: ndarray, logvars: ndarray) -> ndarray:
        return np.array(
            self.__epsilon * tf.exp(logvars * 0.5) + means
        )  # TODO does the sqrt of logvar make sense?

    def __encode(self, x: ndarray) -> Tuple[ndarray, ndarray]:
        x_resized: ndarray = x.reshape(
            (x.shape[0], self.__length * self.__width * self.__channels)
        )
        means, logvars = tf.split(
            self.__encoder(x_resized), num_or_size_splits=2, axis=1
        )
        return np.array(means), np.array(logvars)

    def __decode(self, z: ndarray) -> ndarray:
        images_resized: ndarray = self.__decoder(z).reshape(
            (z.shape[0], self.__length, self.__width, self.__channels)
        )
        return images_resized

    def encode_and_decode(self, x: ndarray) -> ndarray:
        means: ndarray
        logvars: ndarray

        means, logvars = self.__encode(np.array(x))
        z: ndarray = self.__reparameterize(means, logvars)
        x_generated: ndarray = self.__decode(z)

        return np.array(x_generated)

    def save_model(self, path: Optional[str], name: Optional[str]) -> None:
        pass

    def __do_checks_for_init(
        self,
        architecture_encoder: List[int],
        architecture_decoder: List[int],
        learning: float = 0.0001,
        n_distributions: int = 5,
        max_iter: int = 1000,
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

    def __do_checks_for_image_shape(
        self,
        image_length: int = 28,
        image_width: int = 28,
        n_channels: int = 1,
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

    def ___log_normal_pdf(
        self, sample, mean, logvar, raxis=1
    ):  # DKL una manera de implementarlo
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(
            -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis,
        )

    def ___loss(self, x):
        mean, logvar = self.__encode(x)
        z = self.__reparameterize(mean, logvar)
        x_logit = self.__decode(z)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)  # Función de coste entrada-salida
        cross_ent = tf.keras.losses.mean_squared_error(
            x, x_logit
        )  # TODO no es cross ent
        cross_ent = tf.reduce_sum(cross_ent, axis=[1, 2])
        logpz = self.___log_normal_pdf(z, 0.0, 0.0)  # DKL normal
        logqz_x = self.___log_normal_pdf(
            z, mean, logvar
        )  # Tercer cálculo ¿quitar igual?
        return -tf.reduce_mean(
            -cross_ent + logpz - logqz_x
        )  # Hacer un modelo resistente a epsilon?
