from typing import List, Optional, Union, Tuple
import tensorflow as tf
import numpy as np
from Exceptions.illegal_architecture_exception import IllegalArchitectureException
from Domain.VAEModel import VAEModel
from Exceptions.illegal_value_exception import IllegalValueException
from Utils.proyect_typings import Image, NumpyList, GenericList
from Utils.loss_function_selector import LossFunctionSelector
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

"""
Implementation of the most common version of the VAE.
"""


class VAE(VAEModel):
    def __init__(
        self,
        architecture_encoder: GenericList[int],
        architecture_decoder: GenericList[int],
        learning: float = 0.0001,
        n_distributions: int = 5,
        max_iter: int = 1000,
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

        self.__length: Optional[int] = None
        self.__width: Optional[int] = None
        self.__channels: Optional[int] = None
        self.__encoder: Optional[tf.keras.Sequential] = None
        self.__decoder: Optional[tf.keras.Sequential] = None

        self.__encoder_architecture = architecture_encoder
        self.__decoder_architecture = architecture_decoder
        self.__latent: int = n_distributions
        self.__epsilon: List[float] = tf.random.normal(
            shape=(1, self.__latent)
        )  # TODO mirar si esto renta ponerlo en el fit para ajustar el batch

        self.__learning: float = learning
        self.__optimizer = tf.keras.optimizers.Adam(self.__learning)
        self.__iterations: int = max_iter

        # loss_selector = LossFunctionSelector()
        # self.__loss_function = loss_selector.select('DKL_MSE')

    def __train_step(self, x: Image) -> float:
        with tf.GradientTape() as tape:
            # loss: float = self.__loss_function(self, x)
            loss = self.___loss(x)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.__optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return loss

    def __iterate_without_batch(
        self, train_images: GenericList[Image]
    ) -> NumpyList[float]:
        loss_values: List[float] = []

        for iteration in range(1, self.__iterations + 1):
            print(f"Iteration number {iteration}")
            self.generate_and_save_random_images()
            partial_loss: List[float] = []

            for t_x in train_images:
                # TODO ver como hacer para obtener todas las imagenes de una
                partial_loss.append(self.__train_step(t_x))

            loss_values.append(tf.reduce_mean(partial_loss))

        return np.array(loss_values)

    def __iterate_with_normal_batch(
        self, train_images: GenericList[Image]
    ) -> NumpyList[float]:  # TODO adapt
        loss_values: List[float] = []

        for iteration in range(1, self.__iterations + 1):
            print(f"Iteration number {iteration}")
            self.generate_and_save_random_images()
            partial_loss: List[float] = []

            for t_x in train_images:
                partial_loss.append(self.__train_step(t_x))

            loss_values.append(tf.reduce_mean(partial_loss))

        return np.array(loss_values)

    def __iterate_with_cyclic_batch(
        self, train_images: GenericList[Image]
    ) -> NumpyList[float]:  # TODO adapt
        loss_values: List[float] = []

        for iteration in range(1, self.__iterations + 1):
            print(f"Iteration number {iteration}")
            self.generate_and_save_random_images()
            partial_loss: List[float] = []

            for t_x in train_images:
                partial_loss.append(self.__train_step(t_x))

            loss_values.append(tf.reduce_mean(partial_loss))

        return np.array(loss_values)

    def fit_dataset(
        self,
        dataset: Optional[GenericList[int]],
        return_loss: bool = False,
        normalize_pixels: bool = True,
        discretize_pixels: bool = False,
        print_samples: bool = False,
        batch_size: Optional[int] = None,
        batch_is_cyclic: bool = True,
        image_length: int = 28,
        image_width: int = 28,
        n_channels: int = 1,
    ) -> Optional[NumpyList[float]]:
        self.__do_checks_for_image_shape(
            image_length,
            image_width,
            n_channels,
        )

        train_images: GenericList[Image]
        if dataset is None:
            (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
            self.__length = 28
            self.__width = 28
            self.__channels = 1
        else:
            train_images = dataset
            self.__length = image_length
            self.__width = image_width
            self.__channels = n_channels

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

        normalizer: float = 0.0
        if normalize_pixels:
            normalizer = 255.0

        train_images = (
            train_images.reshape(
                (train_images.shape[0], self.__length * self.__width * self.__channels)
            )
            / normalizer
        )  # Cuantas Imágenes x [altura x anchura x nº canales]

        if discretize_pixels:
            train_images = np.where(train_images > 0.5, 1.0, 0.0).astype("float32")

        loss_values: NumpyList[float]
        if batch_size is None:
            loss_values = self.__iterate_without_batch(train_images)
        else:
            if batch_is_cyclic:
                loss_values = self.__iterate_with_cyclic_batch(train_images)
            else:
                loss_values = self.__iterate_with_normal_batch(train_images)

        if return_loss:
            return loss_values

    def __random_sample(self, n_samples: int = 1) -> NumpyList[NumpyList[float]]:
        return np.array(tf.random.normal(shape=(n_samples, self.__latent)))

    def generate_with_random_sample(self, n_samples: int = 1) -> NumpyList[Image]:
        sample: NumpyList[NumpyList[float]] = self.__random_sample(n_samples)
        return np.array(self.__decoder(sample))

    def generate_with_one_sample(
        self, sample: GenericList[float], n_samples: int = 1
    ) -> NumpyList[Image]:
        samples: GenericList[GenericList[float]] = [sample for _ in range(n_samples)]
        return np.array(self.__decoder(samples))

    def generate_with_multiple_samples(
        self, samples: GenericList[GenericList[float]]
    ) -> NumpyList[Image]:
        return np.array(self.__decoder(samples))

    def __reparameterize(
        self, means: NumpyList[float], logvars: NumpyList[float]
    ) -> NumpyList[float]:
        return np.array(
            self.__epsilon * tf.exp(logvars * 0.5) + means
        )  # TODO does the sqrt of logvar make sense?

    def __encode(
        self, x: Image
    ) -> Tuple[NumpyList[float], NumpyList[float]]:  # TODO mirar el resize
        means, logvars = tf.split(self.__encoder(x), num_or_size_splits=2, axis=1)
        return np.array(means), np.array(logvars)

    def __decode(self, z: GenericList[float]) -> Image:  # TODO mirar el resize
        return self.__decoder(z)

    def encode_and_decode(self, x: Image) -> Image:
        means: NumpyList[float]
        logvars: NumpyList[float]

        means, logvars = self.__encode(x)
        z: NumpyList[float] = self.__reparameterize(means, logvars)
        x_generated: Image = self.__decode(z)

        return x_generated

    def __do_checks_for_init(
        self,
        architecture_encoder: GenericList[int],
        architecture_decoder: GenericList[int],
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
        logpz = self.___log_normal_pdf(z, 0.0, 0.0)  # DKL normal
        logqz_x = self.___log_normal_pdf(
            z, mean, logvar
        )  # Tercer cálculo ¿quitar igual?
        return -tf.reduce_mean(
            -cross_ent + logpz - logqz_x
        )  # Hacer un modelo resistente a epsilon?
