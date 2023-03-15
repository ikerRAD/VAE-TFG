from typing import Union, List
import numpy

"""
Typing for the Image objects
"""


class Image:
    def __init__(self):
        self.__name = "Image"

    def __str__(self):
        return self.__name


"""
Typing rename for the numpy ndarray
"""
NumpyList = numpy.ndarray

"""
Typing rename for Union[NumpyList,List]
"""
GenericList = Union[NumpyList, List]

