from abc import abstractmethod, ABC
from typing import List, Optional
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
    def encode_and_decode(self, x: Image) -> Image:
        pass

    @abstractmethod
    def save_model(self, path: Optional[str], name: Optional[str]) -> None:
        pass

    def encode_decode_and_save_images(
        self,
        images: GenericList[Image],
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        m, l = self.encode(x) # TODO adaptar estos
        z = self.reparameterize(m, l)
        image: Image = self.generate(z)[0, :]
        image = image.reshape((1, 28, 28, 1))
        a = x.reshape((1, 28, 28, 1))
        plt.subplot(1, 2, 1)
        plt.imshow(image[0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(a[0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.savefig("image.png")
        plt.show()

    def generate_and_save_random_images(
        self,
        n_images: int = 1,
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        m, l = self.encode(x)
        z = self.reparameterize(m, l)
        image: Image = self.generate(z)[0, :]
        image = image.reshape((1, 28, 28, 1))
        a = x.reshape((1, 28, 28, 1))
        plt.subplot(1, 2, 1)
        plt.imshow(image[0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(a[0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.savefig("image.png")
        plt.show()

    def generate_and_save_images_with_samples(
        self,
        samples: GenericList[GenericList[float]],
        saving_path: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        m, l = self.encode(x)
        z = self.reparameterize(m, l)
        image: Image = self.generate(z)[0, :]
        image = image.reshape((1, 28, 28, 1))
        a = x.reshape((1, 28, 28, 1))
        plt.subplot(1, 2, 1)
        plt.imshow(image[0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(a[0, :, :, 0], cmap="gray")
        plt.axis("off")
        plt.savefig("image.png")
        plt.show()
