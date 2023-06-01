import numpy as np
import tensorflow as tf
from scipy import linalg


def frechet_inception_distance(
    dataset_activations: tf.Tensor, generated_activations: tf.Tensor, eps: float = 1e-6
) -> float:
    assert dataset_activations.shape == generated_activations.shape

    mu_dataset = np.atleast_1d(np.mean(dataset_activations.numpy(), axis=0))
    sigma_dataset = np.atleast_2d(np.cov(dataset_activations.numpy(), rowvar=False))

    mu_generated = np.atleast_1d(np.mean(generated_activations.numpy(), axis=0))
    sigma_generated = np.atleast_2d(np.cov(generated_activations.numpy(), rowvar=False))

    diff = mu_dataset - mu_generated

    covmean, _ = linalg.sqrtm(sigma_dataset.dot(sigma_generated), disp=False)

    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_dataset.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_dataset + offset).dot(sigma_generated + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return max(
        diff.dot(diff)
        + np.trace(sigma_dataset)
        + np.trace(sigma_generated)
        - 2.0 * tr_covmean,
        0,
    )
