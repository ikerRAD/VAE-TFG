from typing import Any, Tuple, Dict, Union

import numpy as np
import tensorflow as tf


def total_variation_distance(
    dataset_labels: tf.Tensor,
    generated_labels: tf.Tensor,
    possible_labels: Any,
    return_distributions: bool = False,
) -> Union[float, Tuple[float, Dict[Any, float], Dict[Any, float]]]:
    assert dataset_labels.shape == generated_labels.shape

    samples_number = dataset_labels.shape[0]

    assert samples_number != 0

    distribution_1 = {}
    distribution_2 = {}

    max_dist = 0.0
    for label in possible_labels:
        distribution_1[label] = (
            np.count_nonzero(
                dataset_labels.numpy()[np.where(dataset_labels.numpy() == label)]
            )
            / samples_number
        )
        distribution_2[label] = (
            np.count_nonzero(
                generated_labels.numpy()[np.where(generated_labels.numpy() == label)]
            )
            / samples_number
        )

        dist = np.abs(distribution_1[label] - distribution_2[label])
        if dist > max_dist:
            max_dist = dist

    if return_distributions is True:
        return max_dist, distribution_1, distribution_2
    return max_dist
