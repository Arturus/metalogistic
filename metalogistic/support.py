import numpy as np


def tupleToDict(tuple):
	return {'cdf_ps': tuple[0], 'cdf_xs': tuple[1], 'lbound': tuple[2], 'ubound': tuple[3]}


def isListLike(object):
	return isinstance(object, list) or (isinstance(object,np.ndarray) and object.ndim==1)


def isNumeric(object):
	return isinstance(object, (float, int, np.int32, np.int64)) or (isinstance(object,np.ndarray) and object.ndim==0)