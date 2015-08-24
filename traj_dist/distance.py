from linecell import trajectory_set_grid

from sspd import e_sspd, g_sspd
from dtw import e_dtw, g_dtw
from erp import e_erp, g_erp
from edr import e_edr, g_edr
from lcss import e_lcss, g_lcss
from frechet import frechet
from discret_frechet import discret_frechet
from hausdorff import e_hausdorff, g_hausdorff
from sowd import sowd_grid,sowd_grid_brut

from c_sspd import c_e_sspd, c_g_sspd
from c_dtw import c_e_dtw, c_g_dtw
from c_erp import c_e_erp, c_g_erp
from c_edr import c_e_edr, c_g_edr
from c_lcss import c_e_lcss, c_g_lcss
from c_hausdorff import c_e_hausdorff, c_g_hausdorff
from c_discret_frechet import c_discret_frechet
from c_frechet import c_frechet
from c_sowd import c_sowd_grid_brut,c_sowd_grid

import numpy as np

import warnings

__all__ = ['distance']

METRIC_DIC = {"geographical": {"cython": {"sspd": c_g_sspd, "dtw": c_g_dtw, "lcss": c_g_lcss, "hausdorff":
    c_g_hausdorff, "sowd_grid": c_sowd_grid, "sowd_grid_brut": c_sowd_grid_brut,"erp" : c_g_erp, "edr" : c_g_edr },
                               "python": {"sspd": g_sspd, "dtw": g_dtw, "lcss": g_lcss, "hausdorff":
                                   g_hausdorff, "sowd_grid": sowd_grid, "sowd_grid_brut": sowd_grid_brut,
                                          "erp" : g_erp, "edr" : g_edr
                               }},
              "euclidean": {"cython": {"sspd": c_e_sspd, "dtw": c_e_dtw, "lcss": c_e_lcss, "hausdorff":
                  c_e_hausdorff, "discret_frechet": c_discret_frechet, "frechet": c_frechet, "sowd_grid": c_sowd_grid,
                                       "sowd_grid_brut": c_sowd_grid_brut, "erp": c_e_erp, "edr": c_e_edr },
                            "python": {"sspd": e_sspd, "dtw": e_dtw, "lcss": e_lcss, "hausdorff":
                                e_hausdorff, "discret_frechet": discret_frechet, "frechet": frechet, "sowd_grid":
                                sowd_grid, "sowd_grid_brut": sowd_grid_brut, "erp" : e_erp, "edr" : e_edr }}}

# ####################
# Pairwise Distance #
# ####################

def pdist(traj_list, metric="sspd", type_d="euclidean", implementation="auto", converted = None, precision = None,
          eps= None, g = None ):
    """
    Usage
    -----
    Pairwise distances between trajectory in traj_list.

    metrics available are :

    1. 'sspd'

        Computes the distances using the Symmetrized Segment Path distance.

    2. 'dtw'

        Computes the distances using the Dynamic Path Warping distance.

    3. 'lcss'

        Computes the distances using the Longuest Common SubSequence distance

    4. 'hausdorf'

        Computes the distances using the Hausdorff distance.

    5. 'frechet'

        Computes the distances using the Frechet distance.

    6. 'discret_frechet'

        Computes the distances using the Discrete Frechet distance.

    7. 'sowd_grid'

        Computes the distances using the Symmetrized One Way Distance.

    8. 'sowd_grid_brut'

        Computes the distances using the Symmetrized Owe Way Distance, brut implementation.

    9. 'erp'

        Computes the distances using the Edit Distance with real Penalty.

    10. 'edr'

        Computes the distances using the Edit Distance on Real sequence.

    type_d available are "euclidean" or "geographical". Some distance can be computing according to geographical space
    instead of euclidean. If so, traj_0 and traj_1 have to be 2-dimensional. First column is longitude, second one
    is latitude.

    If the distance traj_0 and traj_1 are 2-dimensional, the cython implementation is used else the python one is used.
    unless "python" implementation is specified

    'sowd_grid' and sowd_grid_brut', compute distance between trajectory in grid representation. If the coordinate
    are geographical, this conversion can be made according to the geohash encoding. If so, the geohash 'precision'
    is needed.

    'edr' and 'lcss' require 'eps' parameter. These distance assume that two locations are similar, or not, according
    to a given threshold, eps.

    'erp' require g parameter. This distance require a gap parameter. Which must have same dimension that the
    trajectory.

    Parameters
    ----------

    param traj_list:       a list of nT numpy array trajectory
    param metric :         string, distance used
    param type_d :         string, distance type_d used (geographical or euclidean)
    param implementation : string, implementation used (python, cython, auto)
    param converted :      boolean, specified if the data are converted in cell format (sowd_grid and sowd_grid_brut
                           metric)
    param precision :      int, precision of geohash (sowd_grid and sowd_grid_brut)
    param eps :            float, threshold distance (edr and lcss)
    param g :              numpy arrays, gaps (erp distance)


    Returns
    -------

    M : a nT x nT numpy array. Where the i,j entry is the distance between traj_list[i] and traj_list[j]
    """


    list_dim = map(lambda x: x.shape[1] if len(x.shape)>1 else 1, traj_list)
    nb_traj = len(traj_list)
    if not (len(set(list_dim)) == 1):
        raise ValueError("All trajectories must have same dimesion !")
    dim= list_dim[0]

    if not (metric in ["sspd", "dtw", "lcss", "hausdorff", "frechet", "discret_frechet", "sowd_grid",
                       "sowd_grid_brut","erp", "edr"]):
        raise ValueError("The metric argument should be 'sspd', 'dtw', 'lcss','erp','edr' 'hausdorff', 'frechet',"
                         "'discret_frechet', 'sowd_grid' or 'sowd_grid_brut'\nmetric given is : " + metric)

    if not (type_d in ["geographical", "euclidean"]):
        raise ValueError("The type_d argument should be 'euclidean' or 'geographical'\ntype_d given is : " + type_d)

    if not (implementation in ["cython", "python", "auto"]):
        raise ValueError("The implementation argument should be 'cython', 'python' or 'auto'\n implementation given "
                         "is : " + implementation)

    if type_d == "geographical" and (metric in ["frechet", "discret_frechet"]):
        raise ValueError("Geographical implementation for distance "+metric+" is not "
                         "disponible")
    if type_d == "euclidean" and (metric in ["sowd","sowd_grid"]):
        if not(converted):
            raise ValueError("Euclidean implementation for distance "+metric+" is not "
                             "disponible if your data is not already converted in cell format")
    if dim!=2 and implementation == "cython":
        raise ValueError("Implementation with cython is disponible only with 2-dimension trajectories, "
                         "not %d-dimension" %dim)

    if implementation =="auto":
        if dim == 2:
            implementation = "cython"
        else:
            implementation = "python"

    print("Computing " + type_d + " distance " + metric + " with implementation " + implementation + " for %d trajectories" % nb_traj)
    M = np.zeros((nb_traj, nb_traj))
    dist = METRIC_DIC[type_d][implementation][metric]
    if metric.startswith("sowd_grid"):
        if converted is None:
            warnings.warn("converted parameter should be specified for metric sowd_grid and sowd_grid_brut. Default "
                          "is True")
            converted = True
        if converted:
            cells_list=traj_list
        else:
            if precision is None:
                warnings.warn("precision parameter should be specified for metric sowd_grid and sowd_grid_brut if converted "
                      "is False. Default is 7")
                precision = 7
            cells_list_, _, _ =trajectory_set_grid(traj_list,precision)
            cells_list = map(lambda x : x[:,:2],cells_list_)
        for i in range(nb_traj):
            cells_list_i=cells_list[i]
            for j in range(i + 1, nb_traj):
                cells_list_j=cells_list[j]
                M[i, j] = dist(cells_list_i, cells_list_j)
                M[j, i] = M[i, j]
    elif metric == "erp":
        if g is None:
            g = np.zeros(dim,dtype=float)
            warnings.warn("g parameter should be specified for metric erp. Default is ")
            print(g)
        else:
            if g.shape[0]!= dim :
                raise ValueError("g and trajectories in list should have same dimension")

        for i in range(nb_traj):
            traj_list_i = traj_list[i]
            for j in range(i + 1, nb_traj):
                traj_list_j = traj_list[j]
                M[i, j] = dist(traj_list_i, traj_list_j,g)
                M[j, i] = M[i, j]
    elif metric == "lcss" or metric == "edr":
        if eps is None:
            warnings.warn("eps parameter should be specified for metric 'lcss' and 'edr', default is 100 ")
            eps=100
        for i in range(nb_traj):
            traj_list_i = traj_list[i]
            for j in range(i + 1, nb_traj):
                traj_list_j = traj_list[j]
                M[i, j] = dist(traj_list_i, traj_list_j,eps)
                M[j, i] = M[i, j]
    else:
        for i in range(nb_traj):
            traj_list_i = traj_list[i]
            for j in range(i + 1, nb_traj):
                traj_list_j = traj_list[j]
                M[i, j] = dist(traj_list_i, traj_list_j)
                M[j, i] = M[i, j]

    return M

# ########################
#  Distance between list #
# ########################

def cdist(traj_list_1, traj_list_2, metric="sspd", type_d="euclidean", implementation="auto", converted = None, precision = None,
          eps= None, g = None ):
    """
    Usage
    -----
    Computes distance between each pair of the two list of trajectories

    metrics available are :

    1. 'sspd'

        Computes the distances using the Symmetrized Segment Path distance.

    2. 'dtw'

        Computes the distances using the Dynamic Path Warping distance.

    3. 'lcss'

        Computes the distances using the Longuest Common SubSequence distance

    4. 'hausdorf'

        Computes the distances using the Hausdorff distance.

    5. 'frechet'

        Computes the distances using the Frechet distance.

    6. 'discret_frechet'

        Computes the distances using the Discrete Frechet distance.

    7. 'sowd_grid'

        Computes the distances using the Symmetrized One Way Distance.

    8. 'sowd_grid_brut'

        Computes the distances using the Symmetrized Owe Way Distance, brut implementation.

    9. 'erp'

        Computes the distances using the Edit Distance with real Penalty.

    10. 'edr'

        Computes the distances using the Edit Distance on Real sequence.

    type_d available are "euclidean" or "geographical". Some distance can be computing according to geographical space
    instead of euclidean. If so, traj_0 and traj_1 have to be 2-dimensional. First column is longitude, second one
    is latitude.

    If the distance traj_0 and traj_1 are 2-dimensional, the cython implementation is used else the python one is used.
    unless "python" implementation is specified

    'sowd_grid' and sowd_grid_brut', compute distance between trajectory in grid representation. If the coordinate
    are geographical, this conversion can be made according to the geohash encoding. If so, the geohash 'precision'
    is needed.

    'edr' and 'lcss' require 'eps' parameter. These distance assume that two locations are similar, or not, according
    to a given threshold, eps.

    'erp' require g parameter. This distance require a gap parameter. Which must have same dimension that the
    trajectory.

    Parameters
    ----------

    param traj_list:       a list of nT numpy array trajectory
    param metric :         string, distance used
    param type_d :         string, distance type_d used (geographical or euclidean)
    param implementation : string, implementation used (python, cython, auto)
    param converted :      boolean, specified if the data are converted in cell format (sowd_grid and sowd_grid_brut
                           metric)
    param precision :      int, precision of geohash (sowd_grid and sowd_grid_brut)
    param eps :            float, threshold distance (edr and lcss)
    param g :              numpy arrays, gaps (erp distance)


    Returns
    -------

    M : a nT1 x nT2 numpy array. Where the i,j entry is the distance between traj_list_1[i] and traj_list_2[j]

    """

    list_dim_1 = map(lambda x: x.shape[1], traj_list_1)
    nb_traj_1 = len(traj_list_1)
    list_dim_2 = map(lambda x: x.shape[1], traj_list_2)
    nb_traj_2 = len(traj_list_2)
    if not (len(set(list_dim_1 + list_dim_2)) == 1):
        raise ValueError("All trajectories must have same dimesion !")
    dim= list_dim_1[0]

    if not (metric in ["sspd", "dtw", "lcss", "hausdorff", "frechet", "discret_frechet", "sowd_grid",
                       "sowd_grid_brut","erp", "edr"]):
        raise ValueError("The metric argument should be 'sspd', 'dtw', 'lcss','erp','edr' 'hausdorff', 'frechet',"
                         "'discret_frechet', 'sowd_grid' or 'sowd_grid_brut'\nmetric given is : " + metric)

    if not (type_d in ["geographical", "euclidean"]):
        raise ValueError("The type_d argument should be 'euclidean' or 'geographical'\ntype_d given is : " + type_d)

    if not (implementation in ["cython", "python", "auto"]):
        raise ValueError("The implementation argument should be 'cython', 'python' or 'auto'\n implementation given "
                         "is : " + implementation)

    if type_d == "geographical" and (metric in ["frechet", "discret_frechet"]):
        raise ValueError("Geographical implementation for distance "+metric+" is not "
                         "disponible")
    if type_d == "euclidean" and (metric in ["sowd","sowd_grid"]):
        if not(converted):
            raise ValueError("Euclidean implementation for distance "+metric+" is not "
                             "disponible if your data is not already converted in cell format")
    if dim!=2 and implementation == "cython":
        raise ValueError("Implementation with cython is disponible only with 2-dimension trajectories, "
                         "not %d-dimension" %dim)

    if implementation =="auto":
        if dim == 2:
            implementation = "cython"
        else:
            implementation = "python"

    print("Computing " + type_d + " distance " + metric + " with implementation " + implementation + " for %d and %d "
                                                                                                   "trajectories" %
          (nb_traj_1,nb_traj_2))
    M = np.zeros((nb_traj_1, nb_traj_2))
    dist = METRIC_DIC[type_d][implementation][metric]
    if metric.startswith("sowd_grid"):
        if converted is None:
            warnings.warn("converted parameter should be specified for metric sowd_grid and sowd_grid_brut. Default "
                          "is True")
            converted = True
        if converted:
            cells_list_1=traj_list_1
            cells_list_2=traj_list_2
        else:
            if precision is None:
                warnings.warn("precision parameter should be specified for metric sowd_grid and sowd_grid_brut if converted "
                      "is False. Default is 7")
                precision = 7
            cells_list, _, _ =trajectory_set_grid(traj_list_1+traj_list_2,precision)
            cells_list_1 =  map(lambda x : x[:,:2],cells_list[:nb_traj_1])
            cells_list_2 =  map(lambda x : x[:,:2],cells_list[nb_traj_1:])
        for i in range(nb_traj_1):
            cells_list_1_i = cells_list_1[i]
            for j in range(nb_traj_2):
                cells_list_2_j = cells_list_2[j]
                M[i, j] = dist(cells_list_1_i, cells_list_2_j)
    elif metric == "erp":
        if g is None:
            g = np.zeros(dim,dtype=float)
            warnings.warn("g parameter should be specified for metric erp. Default is ")
            print(g)
        else:
            if g.shape[0]!= dim :
                raise ValueError("g and trajectories in list should have same dimension")
        for i in range(nb_traj_1):
            traj_list_1_i = traj_list_1[i]
            for j in range(nb_traj_2):
                traj_list_2_j = traj_list_2[j]
                M[i, j] = dist(traj_list_1_i, traj_list_2_j,g)
    elif metric == "lcss" or metric == "edr":
        if eps is None:
            warnings.warn("eps parameter should be specified for metric 'lcss' and 'edr', default is 100 ")
            eps=100
        for i in range(nb_traj_1):
            traj_list_1_i = traj_list_1[i]
            for j in range(nb_traj_2):
                traj_list_2_j = traj_list_2[j]
                M[i, j] = dist(traj_list_1_i, traj_list_2_j,eps)
    else:
        for i in range(nb_traj_1):
            traj_list_1_i = traj_list_1[i]
            for j in range(nb_traj_2):
                traj_list_2_j = traj_list_2[j]
                M[i, j] = dist(traj_list_1_i, traj_list_2_j)
    return M
