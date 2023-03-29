from dataclasses import dataclass

import tensorflow as tf

"""
Class for the slices inside a batch. The class will only be used internally
"""


@dataclass(frozen=True)
class Slice:
    data: tf.Tensor
