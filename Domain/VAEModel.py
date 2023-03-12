from abc import abstractmethod, ABC
from typing import List
import tensorflow as tf
from Utils.image import Image
import matplotlib.pyplot as plt

"""
Interface for all the VAE and CVAE implementations. The interface follows the a
sklearn-like structure.
"""


class VAEModel(ABC, tf.keras.Model):

    @abstractmethod
    def fit_dataset(self, x: List[Image]) -> List[float]:  # TODO List[Images]
        pass

    @abstractmethod
    def generate_with_random_sample(self) -> Image:
        pass

    @abstractmethod
    def generate(self, sample: List[float]) -> Image:
        pass

    @abstractmethod
    def encode(self, x: Image) -> (List[float], List[float]):
        pass

    @abstractmethod
    def decode(self, z: List[float]) -> Image:
        pass

    @abstractmethod
    def reparametrize(self, means, logvars) -> List[float]:
        pass

    def generate_and_save_image(self, x) -> None:
        m, l = self.encode(x)
        z = self.reparametrize(m, l)
        image: Image = self.generate(z)[0,:]
        image = image.reshape((1,28,28,1))
        a = x.reshape((1,28,28,1))
        plt.subplot(1,2, 1)
        plt.imshow(image[0,:,:,0], cmap='gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(a[0, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig('image.png')
        plt.show()