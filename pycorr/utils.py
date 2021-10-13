"""A few utilities."""

import os
import sys
import time
import logging
import traceback
import functools

import numpy as np

lib_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'lib')


def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '='*100
    #log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level,str):
        level = {'info':logging.INFO,'debug':logging.DEBUG,'warning':logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        def format(self, record):
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ' %(asctime)s %(name)-28s %(levelname)-8s %(message)s'
            return super(MyFormatter,self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename,mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level,handlers=[handler],**kwargs)
    sys.excepthook = exception_handler


class BaseMetaClass(type):

    """Metaclass to add logging attributes to :class:`BaseClass` derived classes."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        cls.set_logger()
        return cls

    def set_logger(cls):
        """
        Add attributes for logging:

        - logger
        - methods log_debug, log_info, log_warning, log_error, log_critical
        """
        cls.logger = logging.getLogger(cls.__name__)

        def make_logger(level):

            @classmethod
            def logger(cls, *args, **kwargs):
                return getattr(cls.logger, level)(*args, **kwargs)

            return logger

        for level in ['debug','info','warning','error','critical']:
            setattr(cls, 'log_{}'.format(level), make_logger(level))


class BaseClass(object,metaclass=BaseMetaClass):
    """
    Base class that implements :meth:`copy`.
    To be used throughout this package.
    """
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, **kwargs):
        new = self.__copy__()
        new.__dict__.update(kwargs)
        return new

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        self.log_info('Saving {}.'.format(filename))
        np.save(filename, self.__getstate__())

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new


def distance(positions):
    """Return cartesian distance, taking coordinates along ``position`` first axis."""
    return np.sqrt(sum(pos**2 for pos in positions))


def cartesian_to_sky(positions, wrap=True, degree=True):
    r"""
    Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    positions : array of shape (3,N)
        Position in cartesian coordinates.

    wrap : bool, default=True
        Whether to wrap RA in :math:`[0, 2 \pi]`.

    degree : bool, default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    dist : array
        Distance.

    ra : array
        Right Ascension.

    dec : array
        Declination.
    """
    dist = distance(positions)
    ra = np.arctan2(positions[1],positions[0])
    if wrap: ra %= 2.*np.pi
    dec = np.arcsin(positions[2]/dist)
    conversion = np.pi/180. if degree else 1.
    return dist, ra/conversion, dec/conversion


def sky_to_cartesian(dist, ra, dec, degree=True, dtype=None):
    """
    Transform distance, RA, Dec into cartesian coordinates.

    Parameters
    ----------
    dist : array
        Distance.

    ra : array
        Right Ascension.

    dec : array
        Declination.

    degree : default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    dtype : numpy.dtype, default=None
        :class:`numpy.dtype` for returned array.

    Returns
    -------
    position : array
        position in cartesian coordinates; of shape (3,len(dist)).
    """
    conversion = 1.
    if degree: conversion = np.pi/180.
    positions = np.empty_like(dist,shape=(3,len(dist)))
    cos_dec = np.cos(dec*conversion)
    positions[0] = dist*cos_dec*np.cos(ra*conversion)
    positions[1] = dist*cos_dec*np.sin(ra*conversion)
    positions[2] = dist*np.sin(dec*conversion)
    return positions


def rebin(ndarray, new_shape, statistic=np.sum):
    """
    Bin an ndarray in all axes based on the target shape, by summing or
    averaging. Number of output dimensions must match number of input dimensions and
    new axes must divide old ones.

    Taken from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    and https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/binned_statistic.html#BinnedStatistic.reindex.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = rebin(m, new_shape=(5,5), statistic=np.sum)
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if ndarray.ndim != len(new_shape):
        raise ValueError('Input array dim is {}, but requested output one is {}'.format(ndarray.ndim, len(new_shape)))

    pairs = []
    for d, c in zip(new_shape, ndarray.shape):
        if c % d != 0:
            raise ValueError('New shape should divide current shape, but {:d} % {:d} = {:d}'.format(c, d, c % d))
        pairs.append((d, c//d))

    flattened = [l for p in pairs for l in p]
    ndarray = ndarray.reshape(flattened)

    for i in range(len(new_shape)):
        ndarray = statistic(ndarray, axis=-1*(i+1))

    return ndarray


# create a lookup table for set bits per byte
_popcount_lookuptable = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)


def popcount(*arrays):
    """
    Return number of 1 bits in each value of input array.
    Inspired from https://github.com/numpy/numpy/issues/16325.
    """
    #if not np.issubdtype(array.dtype, np.unsignedinteger):
    #    raise ValueError('input array must be an unsigned int dtype')
    toret = _popcount_lookuptable[arrays[0].view((np.uint8, (arrays[0].dtype.itemsize,)))].sum(axis=-1)
    for array in arrays[1:]: toret += popcount(array)
    return toret


def pack_bitarrays(*arrays, dtype=np.uint64):
    """
    Pack bit arrays into a list of integer arrays.
    Inverse operation is :func:`unpack_bitarray`, i.e.
    ``unpack_bitarrays(pack_bitarrays(*arrays, dtype=dtype))``is ``arrays``,
    whatever integer ``dtype`` is.

    Parameters
    ----------
    arrays : bool arrays
        Arrays of integers or booleans whose elements should be packed to bits.

    dtype : string, dtype
        Type of output integer arrays.

    Returns
    -------
    arrays : list
        List of integer arrays of type ``dtype``, representing input boolean arrays.
    """
    if not arrays:
        return []
    return reformat_bitarrays(*np.packbits(arrays, axis=0, bitorder='little'), dtype=dtype)


def unpack_bitarrays(*arrays):
    """
    Unpack integer arrays into a bit array.
    Inverse operation is :func:`pack_bitarray`, i.e.
    ``pack_bitarrays(unpack_bitarrays(*arrays), dtype=arrays.dtype)``is ``arrays``.

    Parameters
    ----------
    arrays : integer arrays
        Arrays of integers whose elements should be unpacked to bits.

    Returns
    -------
    arrays : list
        List of boolean arrays of type ``np.uint8``, representing input integer arrays.
    """
    arrayofbytes = reformat_bitarrays(*arrays, dtype=np.uint8)
    return np.unpackbits(arrayofbytes, axis=0, count=None, bitorder='little')


def reformat_bitarrays(*arrays, dtype=np.uint64):
    """
    Reformat input integer arrays into list of arrays of type ``dtype``.
    If, e.g. 6 arrays of type ``np.uint8`` are input, and ``dtype`` is ``np.uint32``,
    a list of 2 arrays is returned.

    Parameters
    ----------
    arrays : integer arrays
        Arrays of integers to reformat.

    dtype : string, dtype
        Type of output integer arrays.

    Returns
    -------
    arrays : list
        List of integer arrays of type ``dtype``, representing input integer arrays.
    """
    dtype = np.dtype(dtype)
    toret = []
    nremainingbytes = 0
    for array in arrays:
        # first bits are in the first byte array
        arrayofbytes = array.view((np.uint8, (array.dtype.itemsize,)))
        arrayofbytes = np.moveaxis(arrayofbytes, -1, 0)
        for arrayofbyte in arrayofbytes:
            if nremainingbytes == 0:
                toret.append([])
                nremainingbytes = dtype.itemsize
            newarray = toret[-1]
            nremainingbytes -= 1
            newarray.append(arrayofbyte[...,None])
    for iarray,array in enumerate(toret):
        npad = dtype.itemsize - len(array)
        if npad: array += [np.zeros_like(array[0])]*npad
        toret[iarray] = np.squeeze(np.concatenate(array,axis=-1).view(dtype), axis=-1)
    return toret


def pascal_triangle(n_rows):
    """
    Compute Pascal triangle.
    Taken from https://stackoverflow.com/questions/24093387/pascals-triangle-for-python.

    Parameters
    ----------
    n_rows : int
        Number of rows in the Pascal triangle, i.e. maximum number of elements :math:`n`.

    Returns
    -------
    triangle : list
        List of list of binomial coefficients.
        The binomial coefficient :math:`(k, n)` is ``triangle[n][k]``.
    """
    toret = [[1]] # a container to collect the rows
    for _ in range(1, n_rows+1):
        row = [1]
        last_row = toret[-1] # reference the previous row
        # this is the complicated part, it relies on the fact that zip
        # stops at the shortest iterable, so for the second row, we have
        # nothing in this list comprension, but the third row sums 1 and 1
        # and the fourth row sums in pairs. It's a sliding window.
        row += [sum(pair) for pair in zip(last_row, last_row[1:])]
        # finally append the final 1 to the outside
        row.append(1)
        toret.append(row) # add the row to the results.
    return toret


def joint_occurences(nrealizations=128, max_occurences=None, case='zerotruncated'):
    """
    Return expected value of inverse counts, i.e. eq. 21 of arXiv:1912.08803.

    Parameters
    ----------
    nrealizations : int
        Number of realizations (including current realization).

    max_occurences : int, default=None
        Maximum number of occurences, less than or equal to ``nrealizations``.
        If ``None``, defaults to ``nrealizations``.

    case : string, default='zerotruncated'
        Refers to the "zero truncated estimator" of arXiv:1912.08803:
        current realization included in bitwise weights and ``nrealizations``.

    Returns
    -------
    occurences : list
        Expected value of inverse counts.
    """
    # gk(c1, c2)
    offset = {'zerotruncated':1, 'all':0}[case]
    if max_occurences is None: max_occurences = nrealizations

    binomial_coeffs = pascal_triangle(nrealizations)

    def prob(c12, c1, c2):
        return binomial_coeffs[c1 - offset][c12 - offset] * binomial_coeffs[nrealizations - c1][c2 - c12] / binomial_coeffs[nrealizations - offset][c2 - offset]

    toret = [[0]] if offset else []
    for c1 in range(offset, max_occurences + 1):
        row = [[0]] if offset else []
        for c2 in range(offset, c1 + 1):
            # we have c12 <= c1, c2 and nrealizations >= c1 + c2 + c12
            row.append(sum(nrealizations / c12 * prob(c12, c1, c2) for c12 in range(max(offset, c1 + c2 - nrealizations), min(c1, c2) + 1)))
        toret.append(row)

    return toret


class DistanceToRedshift(object):

    """Class that holds a conversion distance -> redshift."""

    def __init__(self, distance, zmax=100., nz=2048, interp_order=3):
        """
        Initialize :class:`DistanceToRedshift`.
        Creates an array of redshift -> distance in log(redshift) and instantiates
        a spline interpolator distance -> redshift.

        Parameters
        ----------
        distance : callable
            Callable that provides distance as a function of redshift (array).

        zmax : float, default=100.
            Maximum redshift for redshift <-> distance mapping.

        nz : int, default=2048
            Number of points for redshift <-> distance mapping.

        interp_order : int, default=3
            Interpolation order, e.g. ``1`` for linear interpolation, ``3`` for cubic splines.
        """
        self.distance = distance
        self.zmax = zmax
        self.nz = nz
        zgrid = np.logspace(-8,np.log10(self.zmax),self.nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = self.distance(self.zgrid)
        from scipy import interpolate
        self.interp = interpolate.UnivariateSpline(self.rgrid,self.zgrid,k=interp_order,s=0)

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        return self.interp(distance)
