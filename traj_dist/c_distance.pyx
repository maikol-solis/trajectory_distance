from smowd import e_smowd, g_smowd
from dtw import e_dtw, g_dtw
from lcss import e_lcss, g_lcss
from frechet import frechet
from discret_frechet import discret_frechet
from hausdorff import e_hausdorff, g_hausdorff

from c_smowd import c_e_smowd, c_g_smowd
from c_dtw import c_e_dtw, c_g_dtw
from c_lcss import c_e_lcss, c_g_lcss
from c_hausdorff import c_e_hausdorff, c_g_hausdorff
from c_discret_frechet import c_discret_frechet
from c_frechet import c_frechet

import numpy as np
cimport numpy as np

__all__=['distance']

def c_mat_distance(list traj_list, str dist="smowd",str type="euclidean",dict extra_arg={},str implementation="auto" ):
    """
    Usage
    -----
    Compute the "dist" distance between trajectory traj_0, traj_1.

    dist available are : "smowd", "dtw", "lcss", "hausdorff", "frechet", "discret frechet"

    type available are "euclidean" or "geographical". Some distance can be computing according to geographical space
    instead of euclidean. If so, traj_0 and traj_1 have to be 2-dimensional. First column is longitude, second one
    is latitude.

    If the distance traj_0 and traj_1 are 2-dimensional, the cython implementation is used else the python one is used.
    unless "python" implementation is specified

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param type : string, distance type
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    cdef np.ndarray[np.float64_t,ndim=2] M
    cdef list list_dim
    cdef int nb_traj

    list_dim = map(lambda x : x.shape[1],traj_list)
    nb_traj = len(traj_list)
    if not(np.all(map(lambda x : x==2,list_dim))) :
        raise ValueError("The trajectories must have same dimesion !")
    else:
        if list_dim[0]!=2 or implementation == "python":
            print("Computing "+type+" distance "+dist+" with Python for %d trajectories" %nb_traj)

            if type =="euclidean":
                M=_m_e_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            elif type == "geographical":
                M=_m_g_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            else:
                raise ValueError("type " + type + "Unknown \nShould be geographical or euclidean")
        else :
            print("Computing "+type+" distance "+dist+" with Cython for %d trajectories" %nb_traj)
            if type =="euclidean":
                M=_c_m_e_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            elif type == "geographical":
                M=_c_m_g_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            else:
                raise ValueError("type " + type + "Unknown \nShould be geographical or euclidean")
    return M

cdef np.ndarray[np.float64_t,ndim=2] _m_g_distance(list traj_list,int nb_traj,str dist="smowd",dict extra_arg={}):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "smowd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    cdef np.ndarray[np.float64_t,ndim=2] M
    cdef int i,j

    M=np.zeros((nb_traj,nb_traj))

    if dist == "smowd":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=g_smowd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "dtw":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=g_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=g_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=g_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be smowd, dtw, lcss, hausdorff")
    return M

cdef np.ndarray[np.float64_t,ndim=2] _c_m_g_distance(list traj_list,int nb_traj,str dist="smowd",dict extra_arg={} ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "smowd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    cdef np.ndarray[np.float64_t,ndim=2] M
    cdef int i,j

    M=np.zeros((nb_traj,nb_traj))
    if dist == "smowd":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_g_smowd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "dtw":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_g_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_g_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_g_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be smowd, dtw, lcss, hausdorff")
    return M


cdef np.ndarray[np.float64_t,ndim=2] _m_e_distance(list traj_list,int nb_traj,str dist="smowd",dict extra_arg={} ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "smowd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    cdef np.ndarray[np.float64_t,ndim=2] M
    cdef int i,j

    M=np.zeros((nb_traj,nb_traj))
    if dist == "smowd":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=e_smowd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "dtw":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=e_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=e_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=e_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "frechet":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "discret_frechet":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=discret_frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be smowd, dtw, lcss, hausdorff")
    return M

cdef np.ndarray[np.float64_t,ndim=2] _c_m_e_distance(list traj_list,int nb_traj,str dist="smowd",dict extra_arg={} ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "smowd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    cdef np.ndarray[np.float64_t,ndim=2] M
    cdef int i,j

    M=np.zeros((nb_traj,nb_traj))
    if dist == "smowd":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_e_smowd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]

    elif dist == "dtw":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_e_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_e_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_e_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "frechet":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "discret_frechet":
        for i from 0 <= i < nb_traj:
            for j from i+1 <= j < nb_traj:
                M[i,j]=c_discret_frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be smowd, dtw, lcss, hausdorff")
    return M

