STUFF = "Hi"

from libc.math cimport fmin
from libc.math cimport fmax

cimport numpy as np
import numpy as np

from basic_euclidean import c_eucl_dist

cdef double _c( np.ndarray[np.float64_t,ndim=2] ca,int i, int j, np.ndarray[np.float64_t,ndim=2] P,np.ndarray[np
.float64_t,ndim=2] Q):

    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = c_eucl_dist(P[0,0],P[0,1],Q[0,0],Q[0,1])
    elif i > 0 and j == 0:
        ca[i,j] = fmax(_c(ca,i-1,0,P,Q),c_eucl_dist(P[i,0],P[i,1],Q[0,0],Q[0,1]))
    elif i == 0 and j > 0:
        ca[i,j] = fmax(_c(ca,0,j-1,P,Q),c_eucl_dist(P[0,0],P[0,1],Q[j,0],Q[j,1]))
    elif i > 0 and j > 0:
        ca[i,j] = fmax(fmin(_c(ca,i-1,j,P,Q),fmin(_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q))),c_eucl_dist(P[i,0],P[i,1],
        Q[j,0], Q[j,1]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]

def c_discret_frechet(P,Q):
    """
    Usage
    -----
    Compute the discret frechet distance between trajectories P and Q

    Parameters
    ----------
    param P : px2 array_like, Trajectory P
    param Q : qx2 array_like, Trajectory Q

    Returns
    -------
    frech : float, the discret frechet distance between trajectories P and Q
    """
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca,len(P)-1,len(Q)-1,P,Q)

