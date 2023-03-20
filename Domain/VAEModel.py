from abc import abstractmethod, ABC
from typing import Optional, Union, List
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import ndarray

from Utils.batch_calculators import Batch

"""
Interface for all the VAE and CVAE implementations. The interface follows the a
sklearn-like structure.
"""


class VAEModel(ABC, tf.keras.Model):
    @abstractmethod
    def fit_dataset(
        self,
        return_loss: bool = False,
        batch_size: int = 100,
        batch_type: Optional[Union[str, Batch]] = None,
        generate_images: bool = True,
    ) -> Optional[List[float]]:
        pass

    @abstractmethod
    def generate_with_random_sample(self, n_samples: int = 1) -> tf.Tensor:
        pass

    @abstractmethod
    def generate_with_one_sample(
        self, sample: List[float], n_samples: int = 1
    ) -> tf.Tensor:
        pass

    @abstractmethod
    def generate_with_multiple_samples(self, samples: tf.Tensor) -> tf.Tensor:
        pass

    @abstractmethod
    def encode_and_decode(self, x: tf.Tensor) -> tf.Tensor:
        pass

    def __get_image_path(self, saving_path: Optional[str], name: Optional[str]) -> str:
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

    def __create_figures(
        self, generated_images: ndarray, path: str, image_type: str, save: bool
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
