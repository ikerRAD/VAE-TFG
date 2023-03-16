from abc import abstractmethod, ABC
from typing import Optional
import tensorflow as tf
import matplotlib.pyplot as plt
from Utils.proyect_typings import Image, GenericList, NumpyList

"""
Interface for all the VAE and CVAE implementations. The interface follows the a
sklearn-like structure.
"""


class VAEModel(ABC, tf.keras.Model):
    @abstractmethod
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
        pass

    @abstractmethod
    def generate_with_random_sample(self, n_samples: int = 1) -> NumpyList[Image]:
        pass

    @abstractmethod
    def generate_with_one_sample(
        self, sample: GenericList[float], n_samples: int = 1
    ) -> NumpyList[Image]:
        pass

    @abstractmethod
    def generate_with_multiple_samples(
        self, samples: GenericList[GenericList[float]]
    ) -> NumpyList[Image]:
        pass

    @abstractmethod
    def encode_and_decode(self, x: GenericList[Image]) -> NumpyList[Image]:
        pass

    @abstractmethod
    def save_model(self, path: Optional[str], name: Optional[str]) -> None:
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

    def encode_decode_and_save_images(
        self,
        images: GenericList[Image],
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
        image_type: str = "png",
    ) -> None:
        path: str = self.__get_image_path(saving_path, name)
        generated_images: NumpyList[Image] = self.encode_and_decode(images)

        for i in range(len(images)):
            image: Image = images[i]
            image_generated: Image = generated_images[i]
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(image[0, :, :, :])
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(image_generated[0, :, :, :])
            plt.axis("off")
            plt.savefig(f"{path}_{i + 1}.{image_type}")
            plt.show()

    def __create_figures(
        self, generated_images: NumpyList[Image], path: str, image_type: str
    ) -> None:
        for i in range(len(generated_images)):
            image: Image = generated_images[i]
            plt.figure()
            plt.plot()
            plt.imshow(image[0, :, :, :])
            plt.axis("off")
            plt.savefig(f"{path}_{i + 1}.{image_type}")
            plt.show()

    def generate_and_save_random_images(
        self,
        n_images: int = 1,
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
        image_type: str = "png",
    ) -> None:
        path: str = self.__get_image_path(saving_path, name)
        generated_images: NumpyList[Image] = self.generate_with_random_sample(n_images)

        self.__create_figures(generated_images, path, image_type)

    def generate_and_save_images_with_samples(
        self,
        samples: GenericList[GenericList[float]],
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
        image_type: str = "png",
    ) -> None:
        path: str = self.__get_image_path(saving_path, name)
        generated_images: NumpyList[Image] = self.generate_with_multiple_samples(
            samples
        )

        self.__create_figures(generated_images, path, image_type)
