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
    
cpdef dict forward(np.ndarray[np.int64_t, ndim=2] lattice, np.ndarray[np.float64_t, ndim=3] x_dot_parameters, long S):
    """ Helper to calculate the forward weights.  """
    cdef dict alpha = {}

    cdef unsigned int r
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index
    cdef unsigned int I, J, s

    cdef unsigned int old_i0, old_j0, old_s0 = -1
    cdef np.float64_t edge_potential

    cdef (int, int, int) current_state
    cdef (int, int, int) next_state
    cdef (int, int, int, int, int, int, int) edge

    for r in range(lattice.shape[0]):
        edge = (lattice[r, 0], lattice[r, 1], lattice[r, 2],
                lattice[r, 3], lattice[r, 4], lattice[r, 5],
                lattice[r, 6])
        i0, j0, s0 = current_state = lattice[r, 0], lattice[r, 1], lattice[r, 2]
        next_state = lattice[r, 3], lattice[r, 4], lattice[r, 5]
        edge_parameter_index = edge[6]
        
        if i0 != previous_i0 or j0 != previous_j0 or s0 != previous_s0:
            if current_state[0] == 0 and current_state[1] == 0:
                alpha[current_state] = x_dot_parameters[current_state]
            else:
                alpha[current_state] += x_dot_parameters[current_state]

            previous_i0, previous_j0, previous_s0 = i0, j0, s0

        edge_potential = (x_dot_parameters[next_state[0], next_state[1], edge_parameter_index]
                          + <np.float64_t> alpha[current_state])
        alpha[edge] = edge_potential
        alpha[next_state] = logaddexp(<np.float64_t> alpha.get(next_state, -inf), 
                                      edge_potential)

    I = x_dot_parameters.shape[0] - 1
    J = x_dot_parameters.shape[1] - 1

    for s in range(S):
        if I == J == 0:
            alpha[(I, J, s)] = x_dot_parameters[I, J, s]
        else:
            alpha[(I, J, s)] = <np.float64_t> alpha.get((I, J, s), -inf) + x_dot_parameters[I, J, s]

    return alpha

cpdef np.float64_t[:, :, ::1] forward_predict(np.int64_t[:, ::1] lattice,
                                      np.float64_t[:, :, ::1] x_dot_parameters,
                                      long S) :
    """ Helper to calculate the forward weights for prediction.  """
    cdef np.float64_t[:, :, ::1] alpha = x_dot_parameters.copy()
    alpha[:] = -inf

    cdef unsigned int r 
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index

    cdef int old_s0 = -1

    cdef np.float64_t edge_potential, source_node_potential

    # The lattice is an edgelist matrix where the row are of the form
    #
    # i0, j0, s0, i1, j1, s1, edge_parameter 
    #
    # where (i0, j0, s0) indexes the source node (i1, j1, s1) indexes
    # the target node, and the edge_parameter indicates what "type" of
    # edge this is, i.e. insertion, deletion, substitution
    #
    # i0 and i1 are indices to the first sequence
    # j0 and j1 are indices to the second sequence
    # s0 and s1 are indices to the states
    #
    # The edgelist is sorted by i0, j0, s0, etc. so that
    # edge_parameter is the most quickly varying value and i0 is the
    # least.
    for r in range(lattice.shape[0]):
        i0, j0, s0 = lattice[r, 0], lattice[r, 1], lattice[r, 2]

        if s0 != old_s0 :
            if i0 == 0 and j0 == 0:
                source_node_potential = x_dot_parameters[i0, j0, s0]
            else:
                
                source_node_potential = (alpha[i0,j0,s0]
                                         + x_dot_parameters[i0,j0,s0])
            old_s0 = s0

        i1, j1, s1 = lattice[r, 3], lattice[r, 4], lattice[r, 5]
        edge_parameter_index = lattice[r, 6]

        edge_potential = (x_dot_parameters[i1, j1, edge_parameter_index]
                          + source_node_potential)

        alpha[i1, j1, s1] = logaddexp(alpha[i1, j1, s1], edge_potential)

    cdef int I = alpha.shape[0] - 1
    cdef int J = alpha.shape[1] - 1

    for s in range(S):
        if I == J == 0 :
            alpha[I, J, s] = x_dot_parameters[I, J, s]
        else:
            alpha[I, J, s] += x_dot_parameters[I, J, s]
            
    return alpha


cpdef np.float64_t[:, :, ::1] forward_max_predict(np.int64_t[:, ::1] lattice,
                                            np.float64_t[:, :, ::1] x_dot_parameters,
                                            long S) :
    """ Helper to calculate the forward max-sum weights for prediction.  """

    cdef np.float64_t[:, :, ::1] alpha = x_dot_parameters.copy()
    alpha[:] = -inf

    cdef unsigned int r
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index

    cdef int old_s0 = -1

    cdef np.float64_t edge_potential, source_node_potential

    for r in range(lattice.shape[0]):
        i0, j0, s0 = lattice[r, 0], lattice[r, 1], lattice[r, 2]

        if s0 != old_s0 :
            if i0 == 0 and j0 == 0:
                source_node_potential = x_dot_parameters[i0, j0, s0]
            else:
                source_node_potential = (alpha[i0,j0,s0]
                                         + x_dot_parameters[i0,j0,s0])
            old_s0 = s0

        i1, j1, s1 = lattice[r, 3], lattice[r, 4], lattice[r, 5]
        edge_parameter_index = lattice[r, 6]

        edge_potential = (x_dot_parameters[i1, j1, edge_parameter_index]
                          + source_node_potential)

        alpha[i1, j1, s1] = max(alpha[i1, j1, s1], edge_potential)

    cdef int I = alpha.shape[0] - 1
    cdef int J = alpha.shape[1] - 1

    for s in range(S):
        if I == J == 0 :
            alpha[I, J, s] = x_dot_parameters[I, J, s]
        else:
            alpha[I, J, s] += x_dot_parameters[I, J, s]

    return alpha


cpdef dict backward(ndarray[np.int64_t, ndim=2] lattice,
                    ndarray[np.float64_t, ndim=3] x_dot_parameters,
                    long I, long J, long S):
    """ Helper to calculate the backward weights.  """
    cdef dict beta = {}

    cdef unsigned int r
    cdef unsigned int s
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index

    cdef np.float64_t edge_potential

    for s in range(S):
        beta[(I-1, J-1, s)] = 0.0

    for r in range((lattice.shape[0] - 1), -1, -1):
        i0, j0, s0 = lattice[r, 0], lattice[r, 1], lattice[r, 2], 
        i1, j1, s1 = lattice[r, 3], lattice[r, 4], lattice[r, 5]
        edge_parameter_index = lattice[r, 6]

        edge_potential = <np.float64_t> beta[(i1, j1, s1)] + x_dot_parameters[i1, j1, s1]
        beta[(i0, j0, s0, i1, j1, s1, edge_parameter_index)] = edge_potential
        beta[(i0, j0, s0)] = logaddexp(<np.float64_t> beta.get((i0, j0, s0), -inf),
                                       (edge_potential 
                                        + x_dot_parameters[i1, 
                                                           j1, 
                                                           edge_parameter_index]))
    return beta


def gradient(dict alpha,
             dict beta,
             ndarray[np.float64_t, ndim=2] parameters,
             ndarray[np.int64_t] states_to_classes,
             ndarray[np.float64_t, ndim=3] x,
             long y,
             long I, long J, long K):
    """ Helper to calculate the marginals and from that the gradient given the forward and backward weights. """
    cdef unsigned int n_classes = states_to_classes.max() + 1
    cdef ndarray[np.float64_t] class_Z = np.zeros((n_classes,))
    cdef np.float64_t Z = -inf
    cdef np.float64_t weight
    cdef unsigned int k
    cdef unsigned int state

    for state, clas in enumerate(states_to_classes):
        weight = <np.float64_t> alpha[(I - 1, J - 1, state)]
        class_Z[clas] = weight
        Z = logaddexp(Z, weight)

    cdef ndarray[np.float64_t, ndim=2] derivative = np.full_like(parameters, 0.0)
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index
    cdef np.float64_t alphabeta
    cdef tuple node

    for node in alpha.viewkeys() | beta.viewkeys():
        if len(node) == 3:
            i0, j0, s0 = node
            alphabeta = <np.float64_t>alpha[(i0, j0, s0)] + <np.float64_t>beta[(i0, j0, s0)]

            for k in range(K):
                if states_to_classes[s0] == y:
                    derivative[s0, k] += (exp(alphabeta - class_Z[y]) - exp(alphabeta - Z)) * x[i0, j0, k]
                else:
                    derivative[s0, k] -= exp(alphabeta - Z) * x[i0, j0, k]

        else:
            i0, j0, s0, i1, j1, s1, edge_parameter_index = node
            alphabeta = <np.float64_t>alpha[(i0, j0, s0, i1, j1, s1, edge_parameter_index)] \
                        + <np.float64_t>beta[(i0, j0, s0, i1, j1, s1, edge_parameter_index)]

            for k in xrange(K):
                if states_to_classes[s1] == y:
                    derivative[edge_parameter_index, k] += (exp(alphabeta - class_Z[y]) - exp(alphabeta - Z)) * x[i1, j1, k]
                else:
                    derivative[edge_parameter_index, k] -= exp(alphabeta - Z) * x[i1, j1, k]

    return (class_Z[y]) - (Z), derivative


def gradient_sparse(dict alpha,
                    dict beta,
                    ndarray[np.float64_t, ndim=2] parameters,
                    ndarray[np.int64_t] states_to_classes,
                    ndarray[np.int64_t, ndim=3] x_index,
                    ndarray[np.float64_t, ndim=3] x_value,
                    long y,
                    long I, long J, long K):
    """
    Helper to calculate the marginals and from that the gradient given the forward and backward weights, for
    sparse input features.
    """
    cdef unsigned int n_classes = max(states_to_classes) + 1
    cdef ndarray[np.float64_t] class_Z = np.zeros((n_classes,))
    cdef np.float64_t Z = -inf
    cdef np.float64_t weight
    cdef unsigned int C = K
    cdef unsigned int c
    cdef int k

    for state, clas in enumerate(states_to_classes):
        weight = <np.float64_t> alpha[(I - 1, J - 1, state)]
        class_Z[clas] = weight
        Z = logaddexp(Z, weight)

    cdef ndarray[np.float64_t, ndim=2] derivative = np.full_like(parameters, 0.0)
    cdef unsigned int i0, j0, s0, i1, j1, s1, edge_parameter_index
    cdef np.float64_t alphabeta

    for node in alpha.viewkeys() | beta.viewkeys():
        if len(node) == 3:
            i0, j0, s0 = node
            alphabeta = <np.float64_t>alpha[(i0, j0, s0)] + <np.float64_t>beta[(i0, j0, s0)]

            for c in range(C):
                k = x_index[i0, j0, c]
                if k < 0:
                    break
                if states_to_classes[s0] == y:
                    derivative[s0, k] += (exp(alphabeta - class_Z[y]) - exp(alphabeta - Z)) * x_value[i0, j0, c]
                else:
                    derivative[s0, k] -= exp(alphabeta - Z) * x_value[i0, j0, c]

        else:
            i0, j0, s0, i1, j1, s1, edge_parameter_index = node
            alphabeta = <np.float64_t>alpha[(i0, j0, s0, i1, j1, s1, edge_parameter_index)] \
                                + <np.float64_t>beta[(i0, j0, s0, i1, j1, s1, edge_parameter_index)]

            for c in range(C):
                k = x_index[i1, j1, c]
                if k < 0:
                    break
                if states_to_classes[s1] == y:
                    derivative[edge_parameter_index, k] += (exp(alphabeta - class_Z[y]) - exp(alphabeta - Z)) * x_value[i1, j1, c]
                else:
                    derivative[edge_parameter_index, k] -= exp(alphabeta - Z) * x_value[i1, j1, c]

    return (class_Z[y]) - (Z), derivative


def populate_sparse_features(ndarray[np.float64_t, ndim=3] x,
                             ndarray[np.int64_t, ndim=3] index_array,
                             ndarray[np.float64_t, ndim=3] value_array,
                             long I, long J, long K):
    """ Helper to fill in sparse feature arrays. """
    cdef unsigned int i, j, c, k
    for i in range(I):
        for j in range(J):
            c = 0
            for k in range(K):
                if x[i, j, k] != 0:
                    value_array[i, j, c] = x[i, j, k]
                    index_array[i, j, c] = k
                    c += 1

def sparse_multiply(ndarray[np.float64_t, ndim=3] answer,
                    ndarray[np.int64_t, ndim=3] index_array,
                    ndarray[np.float64_t, ndim=3] value_array,
                    ndarray[np.float64_t, ndim=2] dense_array,
                    long I, long J, long K, long C, long S):
    """ Multiply a sparse three dimensional numpy array (using our own scheme) with a two dimensional array. """
    cdef unsigned int i, j, s, c
    cdef int k
    for i in range(I):
        for j in range(J):
            for s in range(S):
                for c in range(C):
                    k = index_array[i, j, c]
                    if k < 0:
                        break
                    answer[i, j, s] += value_array[i, j, c] * dense_array[k, s]



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
    

