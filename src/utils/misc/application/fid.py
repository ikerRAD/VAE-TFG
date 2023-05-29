from src.project.domain.VAEModel import VAEModel
import tensorflow as tf


class FID:
    def __init__(self, vae_model: VAEModel, dataset: tf.Tensor) -> None:
        pass

    def resample(self) -> None:
        pass

    def calculate_fid(self) -> float:
        pass

    def calculate_solid_fid(self) -> float:
        pass