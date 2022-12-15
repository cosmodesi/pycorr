import numpy as np
cimport numpy as np


cdef extern from "_utils_imp.h":

    float _sum_weights_float(const size_t size1, const size_t size2, float *indweights1, float *indweights2,
                             float *bitweights1, float *bitweights2, const int n_bitwise_weights, const int noffset, const float default_value, const int nthreads)

    double _sum_weights_double(const size_t size1, const size_t size2, double *indweights1, double *indweights2,
                               double *bitweights1, double *bitweights2, const int n_bitwise_weights, const int noffset, const double default_value, const int nthreads)


def sum_weights(indweights1, indweights2, np.ndarray bitweights1, np.ndarray bitweights2, int noffset, double default_value, int nthreads=1):
    dtype = bitweights1.dtype
    cdef size_t size1 = len(bitweights1)
    cdef size_t size2 = len(bitweights2)
    cdef int n_bitwise_weights = bitweights1.shape[1]

    cdef np.ndarray iw1, iw2
    if indweights1 is not None:
        iw1 = np.ascontiguousarray(indweights1)
        iw2 = np.ascontiguousarray(indweights2)
    cdef np.ndarray bw1 = np.ascontiguousarray(bitweights1)
    cdef np.ndarray bw2 = np.ascontiguousarray(bitweights2)

    cdef double * piw1 = NULL, * piw2 = NULL
    cdef float * fpiw1 = NULL, * fpiw2 = NULL

    if dtype.itemsize == 4:
        if indweights1 is not None:
            fpiw1 = <float*> iw1.data
            fpiw2 = <float*> iw2.data
        wsum = _sum_weights_float(size1, size2, fpiw1, fpiw2, <float*> bw1.data, <float*> bw2.data, n_bitwise_weights, noffset, <float> default_value, nthreads)
    else:
        if indweights1 is not None:
            piw1 = <double*> iw1.data
            piw2 = <double*> iw2.data
        wsum = _sum_weights_double(size1, size2, piw1, piw2, <double*> bw1.data, <double*> bw2.data, n_bitwise_weights, noffset, <double> default_value, nthreads)
    return float(wsum)
