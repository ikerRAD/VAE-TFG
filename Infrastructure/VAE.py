from typing import List
import tensorflow as tf
import numpy as np
from Exceptions.illegal_architecture_exception import IllegalArchitectureException
from Domain.VAEModel import VAEModel
from Utils.image import Image
from Utils.loss_function_selector import LossFunctionSelector
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

"""
Implementation of the most common version of the VAE.
"""


class VAE(VAEModel):

    def __init__(self, architecture_encoder: List[int], architecture_decoder: List[int], learning: float = 0.001,
                 n_distributions: int = 5, max_iter: int = -1, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        if len(architecture_encoder) == 0:
            raise IllegalArchitectureException("The architecture of the encoder cannot be empty")

        if len(architecture_decoder) == 0:
            raise IllegalArchitectureException("The architecture of the decoder cannot be empty")

        if architecture_encoder[-1] < n_distributions * 2:
            raise IllegalArchitectureException(
                f"The last layer of the encoder cannot be lower than {n_distributions * 2}")

        if architecture_decoder[0] < n_distributions:
            raise IllegalArchitectureException(
                f"The first layer of the decoder cannot be lower than {n_distributions}")

        self.__latent: int = n_distributions

        self.__encoder = tf.keras.Sequential()
        for n_neurons in architecture_encoder:
            self.__encoder.add(tf.keras.layers.Dense(n_neurons))
        self.__encoder.add(tf.keras.layers.Dense(self.__latent * 2))

        self.__decoder = tf.keras.Sequential()
        self.__decoder.add(tf.keras.layers.InputLayer(input_shape=(self.__latent,)))
        self.__decoder.add(tf.keras.layers.Dense(self.__latent))
        for n_neurons in architecture_decoder:
            self.__decoder.add(tf.keras.layers.Dense(n_neurons))
        self.__decoder.add(tf.keras.layers.Dense(28*28)) #TODO tenerlo en cuenta

        self.__decoder.build((1,28*28))
        self.__encoder.build((1,28*28))

        self.__encoder.summary()

        self.__decoder.summary()

        self.__learning: float = learning
        self.__optimizer = tf.keras.optimizers.Adam(self.__learning)

        self.__iterations: int = max_iter

        self.__epsilon: List[float] = tf.random.normal(shape=(1, self.__latent))

        #loss_selector = LossFunctionSelector()
        #self.__loss_function = loss_selector.select('DKL_MSE')

    def generate_with_random_sample(self) -> Image:
        sample: List[float] = self.random_sample()
        return self.__decoder(sample)

    def random_sample(self) -> List[float]:
        return tf.random.normal(shape=(2,self.__latent))

    def generate(self, sample: List[float]) -> Image:
        return self.__decoder(sample)

    def fit_dataset(self, x: List[Image]) -> List[float]:
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data() #TODO change

        train_images = train_images.reshape((train_images.shape[0], 28 * 28, 1)) / 255.  # Cuantas Imágenes x altura x anchura x nº canales
        train_images = np.where(train_images > .5, 1.0, 0.0).astype('float32')


        for _ in range(0,self.__iterations):
            print(_)
            self.generate_and_save_image(train_images[20])
            for t_x in train_images[0:100]:
                self.__train_step(t_x)


    def __train_step(self, x: Image) -> None:
        with tf.GradientTape() as tape:
            #loss: float = self.__loss_function(self, x)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.__optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def reparametrize(self, means, logvars) -> List[float]:
        return self.__epsilon * tf.exp(logvars * .5) + means

    def encode(self, x: Image) -> (List[float], List[float]):
        means, logvars = tf.split(self.__encoder(x), num_or_size_splits=2, axis=1)
        return means, logvars

    def decode(self, z: List[float]) -> Image:
        return self.__decoder(z)
