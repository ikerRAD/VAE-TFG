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
Typing for a flattened Image objects
"""


class FlattenedImage:
    def __init__(self):
        self.__name = "Image"

    def __str__(self):
        return self.__name


"""
Typing rename for the numpy ndarray
"""
NumpyList = numpy.ndarray

"""
Typing name for a bi-dimensional numpy ndarray
"""
NumpyMatrix = NumpyList[NumpyList]

"""
Typing rename for Union[NumpyList,List]
"""
GenericList = Union[NumpyList, List]

"""
Typing name for any bi-dimensional array
"""
GenericMatrix = Union[NumpyMatrix, List[List]]
