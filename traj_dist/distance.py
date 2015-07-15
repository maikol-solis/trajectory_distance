from sspd import e_sspd, g_sspd
from dtw import e_dtw, g_dtw
from lcss import e_lcss, g_lcss
from frechet import frechet
from discret_frechet import discret_frechet
from hausdorff import e_hausdorff, g_hausdorff
from segment_distance import segments_distance


from c_sspd import c_e_sspd, c_g_sspd
from c_dtw import c_e_dtw, c_g_dtw
from c_lcss import c_e_lcss, c_g_lcss
from c_hausdorff import c_e_hausdorff, c_g_hausdorff
from c_discret_frechet import c_discret_frechet
from c_frechet import c_frechet
from c_segment_distance import c_segments_distance


import numpy as np

__all__=['distance']

def remove_consecutive_point(traj):
    ind_g=np.array(map(lambda x : not(np.all(x)),traj[:-1]==traj[1:]))
    ind=np.where(ind_g)[0]
    return traj[np.hstack((ind_g,True))],ind


def mat_distance(traj_list, dist="sspd", type="euclidean",extra_arg=None,implementation="auto" ):
    """
    Usage
    -----
    Compute the "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff", "frechet", "discret frechet"

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

    list_dim = map(lambda x : x.shape[1],traj_list)
    nb_traj = len(traj_list)
    if not(np.all(map(lambda x : x==2,list_dim))) :
        raise ValueError("The trajectories must have same dimesion !")
    else:
        if list_dim[0]!=2 or implementation == "python":
            print("Computing "+type+" distance "+dist+" with Python for %d trajectories" %nb_traj)

            if type =="euclidean":
                M=m_e_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            elif type == "geographical":
                M=m_g_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            else:
                raise ValueError("type " + type + "Unknown \nShould be geographical or euclidean")
        else :
            print("Computing "+type+" distance "+dist+" with Cython for %d trajectories" %nb_traj)
            if type =="euclidean":
                  M=c_m_e_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            elif type == "geographical":
                M=c_m_g_distance(traj_list,nb_traj, dist=dist,extra_arg=extra_arg )
            else:
                raise ValueError("type " + type + "Unknown \nShould be geographical or euclidean")
    return M

def m_g_distance(traj_list,nb_traj, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    M=np.zeros((nb_traj,nb_traj))

    if dist == "sspd":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=g_sspd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "dtw":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=g_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=g_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=g_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff")
    return M

def c_m_g_distance(traj_list,nb_traj, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    M=np.zeros((nb_traj,nb_traj))
    if dist == "sspd":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_g_sspd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "dtw":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_g_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_g_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_g_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff")
    return M


def m_e_distance(traj_list,nb_traj, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    M=np.zeros((nb_traj,nb_traj))
    if dist == "sspd":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=e_sspd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "dtw":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=e_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=e_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=e_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "frechet":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "discret_frechet":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=discret_frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "segments":
        traj__list=[]
        ind_list=[]
        traj_inds=[]
        for i in range(nb_traj):
            traj_i_,ind_i=remove_consecutive_point(traj_list[i])
            if len(ind_i!=0):
                traj__list.append(traj_i_)
                ind_list.extend(map(lambda x : (i,x),ind_i))
                traj_inds.append(i)
        M=np.zeros((len(ind_list),len(ind_list)))
        ind_list=np.array(ind_list)
        ind_mat=map(lambda x : np.where(ind_list[:,0]==x)[0],traj_inds)
        for ni in traj_inds:
            for nj in traj_inds[ni:]:
                MIJ=segments_distance(traj__list[ni],traj__list[nj])
                M[ind_mat[ni][0]:ind_mat[ni][-1]+1,ind_mat[nj][0]:ind_mat[nj][-1]+1]=MIJ
                if ni!=nj:
                    M[ind_mat[nj][0]:ind_mat[nj][-1]+1,ind_mat[ni][0]:ind_mat[ni][-1]+1]=MIJ.T
        M=(M,ind_list)
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff")
    return M

def c_m_e_distance(traj_list,nb_traj, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    M=np.zeros((nb_traj,nb_traj))
    if dist == "sspd":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_e_sspd(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]

    elif dist == "dtw":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_e_dtw(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "lcss":
        eps = extra_arg["eps"]
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_e_lcss(traj_list[i],traj_list[j],eps)
                M[j,i]=M[i,j]
    elif dist == "hausdorff":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_e_hausdorff(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "frechet":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "discret_frechet":
        for i in range(nb_traj):
            for j in range(i+1,nb_traj):
                M[i,j]=c_discret_frechet(traj_list[i],traj_list[j])
                M[j,i]=M[i,j]
    elif dist == "segments":
        traj__list=[]
        ind_list=[]
        traj_inds=[]
        for i in range(nb_traj):
            traj_i_,ind_i=remove_consecutive_point(traj_list[i])
            if len(ind_i!=0):
                traj__list.append(traj_i_)
                ind_list.extend(map(lambda x : (i,x),ind_i))
                traj_inds.append(i)
        M=np.zeros((len(ind_list),len(ind_list)))
        ind_list=np.array(ind_list)
        ind_mat=map(lambda x : np.where(ind_list[:,0]==x)[0],traj_inds)
        for ni in traj_inds:
            for nj in traj_inds[ni:]:
                MIJ=c_segments_distance(traj__list[ni],traj__list[nj])
                M[ind_mat[ni][0]:ind_mat[ni][-1]+1,ind_mat[nj][0]:ind_mat[nj][-1]+1]=MIJ
                if ni!=nj:
                    M[ind_mat[nj][0]:ind_mat[nj][-1]+1,ind_mat[ni][0]:ind_mat[ni][-1]+1]=MIJ.T
        M=(M,ind_list)

    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff")
    return M

def distance(traj_0, traj_1, dist="sspd", type="euclidean",extra_arg=None,implementation="auto" ):
    """
    Usage
    -----
    Compute the "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff", "frechet", "discret_frechet"

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
    n0= traj_0.shape[1]
    n1= traj_1.shape[1]
    if n0!=n1 :
        raise ValueError("The trajectories must have same dimesion !")
    else:
        if n0!=2 or implementation == "python":
            if type =="euclidean":
                d=e_distance(traj_0, traj_1, dist=dist,extra_arg=extra_arg )
            elif type == "geographical":
                d=g_distance(traj_0, traj_1, dist=dist,extra_arg=extra_arg )
            else:
                raise ValueError("type " + type + "Unknown \nShould be geographical or euclidean")
        else :
            if type =="euclidean":
                d=c_e_distance(traj_0, traj_1, dist=dist,extra_arg=extra_arg )
            elif type == "geographical":
                d=c_g_distance(traj_0, traj_1, dist=dist,extra_arg=extra_arg )
            else:
                raise ValueError("type " + type + "Unknown \nShould be geographical or euclidean")
    return d


def g_distance(traj_0, traj_1, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    if dist == "sspd":
        d=g_sspd(traj_0,traj_1)
    elif dist == "dtw":
        d=g_dtw(traj_0,traj_1)
    elif dist == "lcss":
        eps = extra_arg["eps"]
        d=g_lcss(traj_0,traj_1,eps)
    elif dist == "hausdorff":
        d=g_hausdorff(traj_0,traj_1)
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff")
    return d

def e_distance(traj_0, traj_1, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the euclidean "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff", "frechet", "discret frechet"

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
    if dist == "sspd":
        d=e_sspd(traj_0,traj_1)
    elif dist == "dtw":
        d=e_dtw(traj_0,traj_1)
    elif dist == "lcss":
        eps = extra_arg["eps"]
        d=e_lcss(traj_0,traj_1,eps)
    elif dist == "hausdorff":
        d=e_hausdorff(traj_0,traj_1)
    elif dist == "frechet":
        d=frechet(traj_0,traj_1)
    elif dist == "discret_frechet":
        d=discret_frechet(traj_0,traj_1)
    elif dist == "segments":
        traj_0_,ind_0=remove_consecutive_point(traj_0)
        traj_1_,ind_1=remove_consecutive_point(traj_1)
        d=(segments_distance(traj_0_,traj_1_),ind_0,ind_1)
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff, "
                                           "frechet or discret frechet")
    return d

def c_g_distance(traj_0, traj_1, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the geographical "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff"

    Parameters
    ----------

    param traj_0: len(traj_0) x n numpy array, trajectory
    param traj_1: len(traj_1) x n numpy array, trajectory
    param dist : string, distance used
    param extra_arg : dict, extra argument needeed to some distance

    Returns
    -------
    """
    if dist == "sspd":
        d=c_g_sspd(traj_0,traj_1)
    elif dist == "dtw":
        d=c_g_dtw(traj_0,traj_1)
    elif dist == "lcss":
        eps = extra_arg["eps"]
        d=c_g_lcss(traj_0,traj_1,eps)
    elif dist == "hausdorff":
        d=c_g_hausdorff(traj_0,traj_1)
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff")
    return d

def c_e_distance(traj_0, traj_1, dist="sspd",extra_arg=None ):
    """
    Usage
    -----
    Compute the euclidean "dist" distance between trajectory traj_0, traj_1.

    dist available are : "sspd", "dtw", "lcss", "hausdorff", "frechet", "discret frechet"

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
    if dist == "sspd":
        d=c_e_sspd(traj_0,traj_1)
    elif dist == "dtw":
        d=c_e_dtw(traj_0,traj_1)
    elif dist == "lcss":
        eps = extra_arg["eps"]
        d=c_e_lcss(traj_0,traj_1,eps)
    elif dist == "hausdorff":
        d=c_e_hausdorff(traj_0,traj_1)
    elif dist == "frechet":
        d=c_frechet(traj_0,traj_1)
    elif dist == "discret_frechet":
        d=c_discret_frechet(traj_0,traj_1)
    elif dist == "segments":
        traj_0_,ind_0=remove_consecutive_point(traj_0)
        traj_1_,ind_1=remove_consecutive_point(traj_1)
        d=(c_segments_distance(traj_0_,traj_1_),ind_0,ind_1)
    else:
        raise ValueError("Distance " + dist +" not implemented\n Should be sspd, dtw, lcss, hausdorff, "
                                           "frechet, discret frechet or segments    ")
    return d



