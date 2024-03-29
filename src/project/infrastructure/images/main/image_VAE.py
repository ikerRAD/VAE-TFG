from typing import Optional, List, Union, Tuple, Dict, Callable
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import ndarray
from tensorflow.python.ops.numpy_ops import np_config

from src.project.domain.exceptions.illegal_dataset_exception import (
    IllegalDatasetException,
)
from src.project.domain.exceptions.illegal_value_exception import IllegalValueException
from src.utils.batches.domain.exceptions.no_more_batches_exception import (
    NoMoreBatchesException,
)
from src.project.domain.VAEModel import VAEModel
from src.utils.batches.application.batch_selector import BatchSelector
from src.utils.batches.domain.batch import Batch
from src.utils.epsilons.application.epsilon_generator_selector import (
    EpsilonGeneratorSelector,
)
from src.utils.epsilons.domain.epsilon_generator import EpsilonGenerator

from src.utils.losses.images.application.image_loss_function_selector import (
    ImageLossFunctionSelector,
)

np_config.enable_numpy_behavior()


class ImageVAE(VAEModel):
    def __init__(
        self,
        dataset: Optional[List],
        loss: Union[
            str,
            Callable[
                [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
                Tuple[float, Dict[str, float]],
            ],
        ],
        learning_rate: float,
        n_distributions: int,
        max_iter: int,
        image_height: int,
        image_width: int,
        n_channels: int,
        normalize_data: bool,
        discretize_data: bool,
        *args,
        **kwargs,
    ):
        super().__init__(
            learning_rate,
            n_distributions,
            max_iter,
            *args,
            **kwargs,
        )

        self.__do_checks_for_image_shape(
            image_height,
            image_width,
            n_channels,
        )

        self.__do_loss_checks(loss)

        self._height: int
        self._width: int
        self._channels: int

        if dataset is None:
            (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
            self._height = 28
            self._width = 28
            self._channels = 1
        else:
            train_images = tf.convert_to_tensor(dataset, dtype=tf.float32)
            self._height = image_height
            self._width = image_width
            self._channels = n_channels

        self._train_images = train_images.reshape(
            (train_images.shape[0], self._height, self._width, self._channels)
        )

        normalizer: float = 1.0
        if normalize_data:
            normalizer = 255.0

        self._train_images = self._train_images / normalizer

        if discretize_data:
            self._train_images = tf.cast(
                tf.where(self._train_images > 0.5 * normalizer, 1.0 * normalizer, 0.0),
                tf.float32,
            )
        else:
            self._train_images = tf.cast(self._train_images, tf.float32)

        self._encoder = tf.keras.Sequential()
        self._decoder = tf.keras.Sequential()

        self._epsilon: Optional[EpsilonGenerator] = None

        self._loss_function = ImageLossFunctionSelector.select(loss)

    def fit_dataset(
        self,
        return_loss: bool = False,
        epsilon_generator: Union[
            str, EpsilonGenerator
        ] = EpsilonGeneratorSelector.possible_keys()[0],
        batch_size: int = 100,
        batch_type: Optional[Union[str, Batch]] = BatchSelector.possible_keys()[0],
        generate_samples: bool = True,
        sample_frequency: int = 10,
    ) -> Optional[Tuple[List[float], List[Dict[str, float]]]]:
        loss_values: List[float]
        loss_summaries: List[Dict[str, float]]

        self.__do_checks_for_epsilon_and_batch(
            epsilon_generator, batch_size, batch_type
        )

        self._epsilon = EpsilonGeneratorSelector.select(epsilon_generator)

        if batch_type is None:
            self._epsilon.set_up(self._train_images.shape[0], self._latent)

            loss_values, loss_summaries = self._iterate_without_batch(
                self._train_images, generate_samples, sample_frequency
            )
        else:
            self._epsilon.set_up(batch_size, self._latent)

            the_batch: Batch = BatchSelector.select(batch_type)

            the_batch.set_up(self._train_images, batch_size)
            loss_values, loss_summaries = self._iterate_with_batch(
                the_batch, generate_samples, sample_frequency
            )

        if return_loss:
            return loss_values, loss_summaries

    def _encode(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        means, logvars = tf.split(self._encoder(x), num_or_size_splits=2, axis=1)
        return means, logvars

    def _reparameterize(self, means: tf.Tensor, logvars: tf.Tensor) -> tf.Tensor:
        return self._epsilon.get() * tf.exp(logvars * tf.constant(0.5)) + means

    def _decode(self, z: tf.Tensor) -> tf.Tensor:
        images_resized: tf.Tensor = self._decoder(z).reshape(
            (z.shape[0], self._height, self._width, self._channels)
        )
        return images_resized

    def encode_and_decode(self, x: tf.Tensor) -> tf.Tensor:
        means: tf.Tensor
        logvars: tf.Tensor

        means, logvars = self._encode(x)
        z: tf.Tensor = self._reparameterize(means, logvars)
        x_generated: tf.Tensor = self._decode(z)

        return x_generated

    @tf.function
    @tf.autograph.experimental.do_not_convert
    def _train_step(self, x: tf.Tensor) -> Tuple[float, Dict[str, float]]:
        with tf.GradientTape() as tape:
            means: tf.Tensor
            logvars: tf.Tensor
            means, logvars = self._encode(x)
            z: tf.Tensor = self._reparameterize(means, logvars)
            x_generated: tf.Tensor = self._decode(z)

            loss: float
            loss_summary: Dict[str, float]
            loss, loss_summary = self._loss_function(z, means, logvars, x, x_generated)

            gradients = tape.gradient(loss, self.trainable_variables)
            self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            return loss, loss_summary

    def _iterate_without_batch(
        self,
        train_images: tf.Tensor,
        generate_images: bool,
        sample_frequency: int,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        loss_values: List[float] = []
        loss_summaries: List[Dict[str, float]] = []

        partial_loss: float
        partial_summary: Dict[str, float]
        for iteration in range(1, self._iterations + 1):
            # print(f"Iteration number {iteration}")
            if generate_images:
                if iteration % sample_frequency == 0:
                    self.generate_random_images(save=False)

            partial_loss, partial_summary = self._train_step(train_images)

            loss_values.append(partial_loss)
            loss_summaries.append(partial_summary)

        return loss_values, loss_summaries

    def _iterate_with_batch(
        self,
        the_batch: Batch,
        generate_images: bool,
        sample_frequency: int,
    ) -> Tuple[List[float], List[Dict[str, float]]]:
        loss_values: List[float] = []
        loss_summaries: List[Dict[str, float]] = []

        partial_loss: float
        partial_summary: Dict[str, float]

        train_images: tf.Tensor

        try:
            for iteration in range(1, self._iterations + 1):
                # print(f"Iteration number {iteration}")
                if generate_images:
                    if iteration % sample_frequency == 0:
                        self.generate_random_images(save=False)

                train_images = the_batch.next()
                partial_loss, partial_summary = self._train_step(train_images)

                loss_values.append(partial_loss)
                loss_summaries.append(partial_summary)

        except NoMoreBatchesException as termination:
            print(str(termination))

        return loss_values, loss_summaries

    def get_image_shape(self) -> Tuple[int, int, int]:
        return self._height, self._width, self._channels

    def change_dataset(
        self,
        dataset: tf.Tensor,
        normalize_data: bool = True,
        discretize_data: bool = False,
    ) -> None:
        train_images = tf.convert_to_tensor(dataset, dtype=tf.float32)

        if train_images.shape[1:] != self.get_image_shape():
            raise IllegalDatasetException(
                f"The given data has the shape {tuple(train_images.shape[1:])},"
                f" and it should be {self.get_image_shape()}"
            )

        normalizer: float = 1.0
        if normalize_data:
            normalizer = 255.0
        self._train_images = train_images / normalizer
        if discretize_data:
            self._train_images = tf.cast(
                tf.where(self._train_images > 0.5 * normalizer, 1.0 * normalizer, 0.0),
                tf.float32,
            )

    def add_train_instances(
        self,
        instances: tf.Tensor,
        normalize_data: bool = True,
        discretize_data: bool = False,
        shuffle_data: bool = False,
    ) -> None:
        train_images = tf.convert_to_tensor(instances, dtype=tf.float32)

        if train_images.shape[1:] != self.get_image_shape():
            raise IllegalDatasetException(
                f"The given data has the shape {tuple(train_images.shape[1:])},"
                f" and it should be {self.get_image_shape()}"
            )

        self._train_images = tf.concat(
            [
                self._train_images,
                instances,
            ],
            axis=0,
        )

        if shuffle_data:
            self._train_images = tf.random.shuffle(self._train_images)

    def encode_decode_images(
        self,
        images: tf.Tensor,
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
        image_type: str = "png",
        save: bool = True,
    ) -> None:
        np_images: ndarray = images.numpy()
        path: str = self.__get_image_path(saving_path, name)
        generated_images: ndarray = self.encode_and_decode(images).numpy()

        for i in range(len(images)):
            image: ndarray = np_images[i]
            image_generated: ndarray = generated_images[i]
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap="gray")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(image_generated[0, :, :, :])
            plt.axis("off")
            if save:
                plt.savefig(f"{path}_{i + 1}.{image_type}")
            plt.show()

    def generate_random_images(
        self,
        n_images: int = 1,
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
        image_type: str = "png",
        save: bool = True,
    ) -> None:
        path: str = self.__get_image_path(saving_path, name)
        generated_images: ndarray = self.generate_with_random_sample(n_images).numpy()

        self.__create_figures(generated_images, path, image_type, save)

    def generate_images_with_samples(
        self,
        samples: tf.Tensor,
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
        image_type: str = "png",
        save: bool = True,
    ) -> None:
        path: str = self.__get_image_path(saving_path, name)
        generated_images: ndarray = self.generate_with_multiple_samples(samples).numpy()

        self.__create_figures(generated_images, path, image_type, save)

    @staticmethod
    def __create_figures(
        generated_images: ndarray, path: str, image_type: str, save: bool
    ) -> None:
        for i in range(len(generated_images)):
            image: ndarray = generated_images[i]
            plt.figure()
            plt.plot()
            plt.imshow(image, cmap="gray")
            plt.axis("off")
            if save:
                plt.savefig(f"{path}_{i + 1}.{image_type}")
            plt.show()

    @staticmethod
    def __get_image_path(saving_path: Optional[str], name: Optional[str]) -> str:
        path: str
        if saving_path is None:
            path = ""
        else:
            if saving_path[-1] != "/":
                path = saving_path + "/"
            else:
                path = saving_path

        if name is None:
            path = path + "unnamed"
        else:
            path = path + name

        return path

    @staticmethod
    def __do_checks_for_image_shape(
        image_height,
        image_width,
        n_channels,
    ) -> None:
        if image_height <= 0:
            raise IllegalValueException(
                "The 'image height' parameter cannot be less than or equal to zero."
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

    @staticmethod
    def __do_loss_checks(
        loss: Union[
            str,
            Callable[
                [tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
                Tuple[float, Dict[str, float]],
            ],
        ]
    ) -> None:
        if type(loss) is str:
            if loss not in ImageLossFunctionSelector.possible_keys():
                raise IllegalValueException(
                    f"The loss function with the key '{loss}' does not exist"
                )

    @staticmethod
    def __do_checks_for_epsilon_and_batch(
        epsilon_generator: Union[str, EpsilonGenerator],
        batch_size: int,
        batch_type: Optional[Union[str, Batch]],
    ) -> None:
        if type(epsilon_generator) is str:
            if epsilon_generator not in EpsilonGeneratorSelector.possible_keys():
                raise IllegalValueException(
                    f"The epsilon generator with the key '{epsilon_generator}' does not exist"
                )

        if batch_size <= 0:
            raise IllegalValueException(
                f"The batch size cannot be lower than 1, got {batch_size}"
            )

        if type(batch_type) is str:
            if batch_type not in BatchSelector.possible_keys():
                raise IllegalValueException(
                    f"The batch with the key '{batch_type}' does not exist"
                )
