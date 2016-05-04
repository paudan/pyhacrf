#cython: boundscheck=False, wraparound=False, initializedcheck=False
#cython: infer_types=True

import numpy as np
cimport numpy as np
from numpy import ndarray
from numpy cimport ndarray
from numpy.math cimport INFINITY as inf
cdef extern from "fastlogexp.h" nogil :
    np.float64_t log "fastlog" (np.float64_t x)
    np.float64_t exp "fastexp" (np.float64_t x)


cpdef dict forward(np.ndarray[np.float64_t, ndim=3] x_dot_parameters, int S):
    """ Helper to calculate the forward weights.  """
    cdef dict alpha = {}

    cdef int I, J
    I, J = x_dot_parameters.shape[0], x_dot_parameters.shape[1]

    # Fill in the edges of the state matrices
    #
    #   0 1 2 3 
    # 0 x x x x
    # 1 x - - -
    # 2 x - - -
    # 3 x - - -
    cdef int insertion = 0
    cdef int deletion = 1
    cdef int matching = 2
    cdef np.float64_t insert, delete, match
    cdef int i, j, s 

    for s in range(S):
        alpha[0, 0, s] = x_dot_parameters[0, 0, s]
        for i in range(1, I):
            insert = (alpha[i - 1, 0, s] +
                      x_dot_parameters[i, 0, insertion + s])
            alpha[i, 0, s] = x_dot_parameters[i, 0, s] + insert

            alpha[i - 1, 0, s, i, 0, s, insertion + s] = insert
        for j in range(1, J):
            delete = (alpha[0, j - 1, s] +
                      x_dot_parameters[0, j, deletion + s])
            alpha[0, j, s] = x_dot_parameters[0, j, s] + delete

            alpha[0, j - 1, s, 0, j, s, deletion + s] = delete
        
        # Now fill in the middle of the matrix    
        for i in range(1, I):
            for j in range(1, J):
                insert = (alpha[i - 1, j, s] +
                          x_dot_parameters[i, j, insertion + s])
                delete = (alpha[i, j - 1, s] +
                          x_dot_parameters[i, j, deletion + s])
                match = (alpha[i - 1, j - 1, s] +
                         x_dot_parameters[i, j, matching + s])
                alpha[i, j, s] = (x_dot_parameters[i, j, s] +
                                  logaddexp(insert, logaddexp(delete, match)))

                alpha[i - 1, j, s, i, j, s, insertion + s] = insert
                alpha[i, j - 1, s, i, j, s, insertion + s] = delete
                alpha[i - 1, j - 1, s, i, j, s, insertion + s] = match

    return alpha


cpdef np.float64_t[:, :, ::1] forward_predict(np.float64_t[:, :, ::1] x_dot_parameters,
                                              int S) :
    cdef np.float64_t[:, :, ::1] alpha = x_dot_parameters.copy()

    cdef int I, J
    I, J = alpha.shape[0], alpha.shape[1]

    # Fill in the edges of the state matrices
    #
    #   0 1 2 3 
    # 0 x x x x
    # 1 x - - -
    # 2 x - - -
    # 3 x - - -
    cdef int insertion = 0
    cdef int deletion = 1
    cdef int matching = 2
    cdef np.float64_t insert, delete, match
    cdef int i, j, s
    
    for s in range(S):
        alpha[0, 0, s] = x_dot_parameters[0, 0, s]
        for i in range(1, I):
            insert = (alpha[i - 1, 0, s] +
                      x_dot_parameters[i, 0, insertion + s])
            alpha[i, 0, s] = x_dot_parameters[i, 0, s] + insert
        for j in range(1, J):
            delete = (alpha[0, j - 1, s] +
                      x_dot_parameters[0, j, deletion + s])
            alpha[0, j, s] = x_dot_parameters[0, j, s] + delete
        
        # Now fill in the middle of the matrix    
        for i in range(1, I):
            for j in range(1, J):
                insert = (alpha[i - 1, j, s] +
                          x_dot_parameters[i, j, insertion + s])
                delete = (alpha[i, j - 1, s] +
                          x_dot_parameters[i, j, deletion + s])
                match = (alpha[i - 1, j - 1, s] +
                         x_dot_parameters[i, j, matching + s])
                alpha[i, j, s] = (x_dot_parameters[i, j, s] +
                                  logaddexp(insert, logaddexp(delete, match)))

    return alpha

cdef np.float64_t logaddexp(np.float64_t x, np.float64_t y) nogil :
    cdef np.float64_t tmp
    if x == y :
        return x + log(2)
    else :
        tmp = x - y
        if tmp > 0 :
            return x + log(1 + exp(-tmp))
        elif tmp <= 0 :
            return y + log(1 + exp(tmp))
        else :
            return tmp
    
