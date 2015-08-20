#cython: boundscheck=False
#cython: wraparound=False

from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport acos
from libc.math cimport asin
from libc.math cimport atan2
from libc.math cimport sqrt
from libc.math cimport fmin
from libc.math cimport fmax
from libc.math cimport fabs

from cpython cimport bool
cimport numpy as np
from numpy.math cimport INFINITY

cdef float pi = 3.14159265
cdef float rad = pi/180.0
cdef int R = 6378137





###############
### FRECHET ###
###############

cdef np.ndarray[np.float64_t,ndim=2] c_point_line_intersection(double px, double py, double s1x, double s1y,
double
s2x, double s2y, double eps):
    """
    Usage
    -----
    Find the intersections between the circle of radius eps and center (px, py) and the line delimited by points
    (s1x, s1y) and (s2x, s2y).
    It is supposed here that the intersection between them exists. If no, raise error

    Parameters
    ----------
    param px : float, centre's abscissa of the circle
    param py : float, centre's ordinate of the circle
    eps : float, radius of the circle
    s1x : abscissa of the first point of the line
    s1y : ordinate of the first point of the line
    s2x : abscissa of the second point of the line
    s2y : ordinate of the second point of the line

    Returns
    -------
    intersect : 2x2 numpy_array
                Coordinate of the two intersections.
                If the two intersections are the same, that's means that the line is a tangent of the circle.
    """
    cdef double rac,y1,y2,m,c,A,B,C,delta,x,y,sdelta,x1,x2
    cdef np.ndarray[np.float64_t,ndim=2] intersect
    cdef bool delta_strict_inf_0,delta_inf_0,delta_strict_sup_0

    if s2x==s1x :
        rac=sqrt((eps*eps) - ((s1x-px)*(s1x-px)))
        y1 = py+rac
        y2 = py-rac
        intersect = np.array([[s1x,y1],[s1x,y2]])
    else:
        m= (s2y-s1y)/(s2x-s1x)
        c= s2y-m*s2x
        A=m*m+1
        B=2*(m*c-m*py-px)
        C=py*py-eps*eps+px*px-2*c*py+c*c
        delta=B*B-4*A*C
        delta_inf_0 = delta <= 0
        delta_strict_sup_0 = delta > 0
        if delta_inf_0 :
            delta_strict_inf_0 = delta <0

            if delta_strict_inf_0 :
                print(delta)
            x = -B/(2*A)
            y = m*x+c
            intersect = np.array([[x,y],[x,y]])
        elif delta_strict_sup_0 > 0 :
            sdelta = sqrt(delta)
            x1= (-B+sdelta)/(2*A)
            y1=m*x1+c
            x2= (-B-sdelta)/(2*A)
            y2=m*x2+c
            intersect = np.array([[x1,y1],[x2,y2]])
        else :
            raise ValueError("The intersection between circle and line is supposed to exist")
    return intersect

def point_line_intersection(double px, double py, double s1x, double s1y, double
s2x, double s2y, double eps):
    """
    Usage
    -----
    Find the intersections between the circle of radius eps and center (px, py) and the line delimited by points
    (s1x, s1y) and (s2x, s2y).
    It is supposed here that the intersection between them exists. If no, raise error

    Parameters
    ----------
    param px : float, centre's abscissa of the circle
    param py : float, centre's ordinate of the circle
    param eps : float, radius of the circle
    param s1x : abscissa of the first point of the line
    param s1y : ordinate of the first point of the line
    param s2x : abscissa of the second point of the line
    param s2y : ordinate of the second point of the line

    Returns
    -------
    intersect : 2x2 numpy_array
                Coordinate of the two intersections.
                If the two intersections are the same, that's means that the line is a tangent of the circle.
    """
    cdef double rac,y1,y2,m,c,A,B,C,delta,x,y,sdelta,x1,x2
    cdef np.ndarray[np.float64_t,ndim=2] intersect
    cdef bool delta_strict_inf_0,delta_inf_0,delta_strict_sup_0

    if s2x==s1x :
        rac=sqrt((eps*eps) - ((s1x-px)*(s1x-px)))
        y1 = py+rac
        y2 = py-rac
        intersect = np.array([[s1x,y1],[s1x,y2]])
    else:
        m= (s2y-s1y)/(s2x-s1x)
        c= s2y-m*s2x
        A=m*m+1
        B=2*(m*c-m*py-px)
        C=py*py-eps*eps+px*px-2*c*py+c*c
        delta=B*B-4*A*C
        delta_inf_0 = delta <= 0
        delta_strict_sup_0 = delta > 0
        if delta_inf_0 :
            delta_strict_inf_0 = delta <0

            if delta_strict_inf_0 :
                print(delta)
            x = -B/(2*A)
            y = m*x+c
            intersect = np.array([[x,y],[x,y]])
        elif delta_strict_sup_0 > 0 :
            sdelta = sqrt(delta)
            x1= (-B+sdelta)/(2*A)
            y1=m*x1+c
            x2= (-B-sdelta)/(2*A)
            y2=m*x2+c
            intersect = np.array([[x1,y1],[x2,y2]])
        else :
            raise ValueError("The intersection between circle and line is supposed to exist")
    return intersect


cdef np.ndarray[np.float64_t,ndim=1] c_free_line(double px, double py, double eps, double s1x, double s1y, double
s2x, double s2y):
    """
    Usage
    -----
    Return the free space in the segment s, from point p.
    This free space is the set of all point in s whose distance from p is at most eps.
    Since s is a segment, the free space is also a segment.
    We return a 1x2 array whit the fraction of the segment s which are in the free space.
    If no part of s are in the free space, return [-1,-1]

    Parameters
    ----------
    param px : float, centre's abscissa of the circle
    param py : float, centre's ordinate of the circle
    param eps : float, radius of the circle
    param s1x : abscissa of the first end point of the segment
    param s1y : ordinate of the first end point of the segment
    param s2x : abscissa of the second end point of the segment
    param s2y : ordinate of the second end point of the segment

    Returns
    -------
    lf : 1x2 numpy_array
         fraction of segment which is in the free space (i.e [0.3,0.7], [0.45,1], ...)
         If no part of s are in the free space, return [-1,-1]
    """
    cdef np.ndarray[np.float64_t,ndim=2] intersect
    cdef np.ndarray[np.float64_t,ndim=1] lf,ordered_point
    cdef double i1x,i1y,u1,i2x,i2y,u2,segl,segl2
    cdef bool pts_sup_eps,i1x_dif_i2x,i1y_dif_i2y,u1_sup_0,u2_inf_1

    pts_sup_eps = point_to_seg(px,py,s1x,s1y,s2x,s2y)>eps
    if pts_sup_eps:
        #print("No Intersection")
        lf=np.array([-1.0,-1.0])
    else :
        segl=eucl_dist(s1x,s1y,s2x,s2y)
        segl2=segl*segl
        intersect = c_point_line_intersection(px,py,s1x,s1y,s2x,s2y,eps)
        i1x = intersect[0,0]
        i2x = intersect[1,0]
        i1y = intersect[0,1]
        i2y = intersect[1,1]
        i1x_dif_i2x = i1x!=i2x
        i1y_dif_i2y = i1y!=i2y
        if i1x_dif_i2x or i1y_dif_i2y:
            u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y)))/segl2
            u2 = (((i2x - s1x) * (s2x - s1x)) + ((i2y - s1y) * (s2y - s1y)))/segl2
            ordered_point=np.array(sorted((0,1,u1,u2)))
            lf= ordered_point[1:3]
        else :
            u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y)))/segl2
            u1_sup_0 = u1 >=0
            u2_inf_1 = u1 <=1
            if u1_sup_0 and u2_inf_1:
                lf=np.array([u1,u1])
            else:
                lf=np.array([-1.0,-1.0])
    return lf


def free_line(double px, double py, double eps, double s1x, double s1y, double s2x, double s2y):
    """
    Usage
    -----
    Return the free space in the segment s, from point p.
    This free space is the set of all point in s whose distance from p is at most eps.
    Since s is a segment, the free space is also a segment.
    We return a 1x2 array whit the fraction of the segment s which are in the free space.
    If no part of s are in the free space, return [-1,-1]

    Parameters
    ----------
    param px : float, centre's abscissa of the circle
    param py : float, centre's ordinate of the circle
    param eps : float, radius of the circle
    param s1x : abscissa of the first end point of the segment
    param s1y : ordinate of the first end point of the segment
    param s2x : abscissa of the second end point of the segment
    param s2y : ordinate of the second end point of the segment

    Returns
    -------
    lf : 1x2 numpy_array
         fraction of segment which is in the free space (i.e [0.3,0.7], [0.45,1], ...)
         If no part of s are in the free space, return [-1,-1]
    """
    cdef np.ndarray[np.float64_t,ndim=2] intersect
    cdef np.ndarray[np.float64_t,ndim=1] lf,ordered_point
    cdef double i1x,i1y,u1,i2x,i2y,u2,segl,segl2
    cdef bool pts_sup_eps,i1x_dif_i2x,i1y_dif_i2y,u1_sup_0,u2_inf_1

    pts_sup_eps = point_to_seg(px,py,s1x,s1y,s2x,s2y)>eps
    if pts_sup_eps:
        #print("No Intersection")
        lf=np.array([-1.0,-1.0])
    else :
        segl=eucl_dist(s1x,s1y,s2x,s2y)
        segl2=segl*segl
        intersect = c_point_line_intersection(px,py,s1x,s1y,s2x,s2y,eps)
        i1x = intersect[0,0]
        i2x = intersect[1,0]
        i1y = intersect[0,1]
        i2y = intersect[1,1]
        i1x_dif_i2x = i1x!=i2x
        i1y_dif_i2y = i1y!=i2y
        if i1x_dif_i2x or i1y_dif_i2y:
            u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y)))/segl2
            u2 = (((i2x - s1x) * (s2x - s1x)) + ((i2y - s1y) * (s2y - s1y)))/segl2
            ordered_point=np.array(sorted((0,1,u1,u2)))
            lf= ordered_point[1:3]
        else :
            u1 = (((i1x - s1x) * (s2x - s1x)) + ((i1y - s1y) * (s2y - s1y)))/segl2
            u1_sup_0 = u1 >=0
            u2_inf_1 = u1 <=1
            if u1_sup_0 and u2_inf_1:
                lf=np.array([u1,u1])
            else:
                lf=np.array([-1.0,-1.0])
    return lf

cdef c_compute_LF_BF(np.ndarray[np.float64_t,ndim=2] P, np.ndarray[np.float64_t,ndim=2] Q, int p, int q,
double eps):
    """
    Usage
    -----
    Compute all the free space on the boundary of cells in the diagram for polygonal chains P and Q and the given eps
    LF[(i,j)] is the free space of segments [Pi,Pi+1] from point  Qj
    BF[(i,j)] is the free space of segment [Qj,Qj+1] from point Pj

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance

    Returns
    -------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    """

    cdef dict LF,BF
    cdef int j,i
    cdef double Q_j0,Q_j1,Q_j10,Q_j11,P_i0,P_i1,P_i10,P_i11

    LF={}
    for j from 0 <= j < q:
        for i from 0 <= i < p-1 :
            Q_j0=Q[j,0]
            Q_j1=Q[j,1]
            P_i0=P[i,0]
            P_i1=P[i,1]
            P_i10=P[i+1,0]
            P_i11=P[i+1,1]

            LF.update({(i,j):c_free_line(Q_j0,Q_j1,eps,P_i0,P_i1,P_i10,P_i11)})
    BF={}
    for j from 0 <= j < q-1:
        for i from 0 <= i < p :
            Q_j0=Q[j,0]
            Q_j1=Q[j,1]
            Q_j10=Q[j+1,0]
            Q_j11=Q[j+1,1]
            P_i0=P[i,0]
            P_i1=P[i,1]

            BF.update({(i,j):c_free_line(P_i0,P_i1,eps,Q_j0,Q_j1,Q_j10,Q_j11)})
    return LF,BF

def compute_LF_BF(np.ndarray[np.float64_t,ndim=2] P, np.ndarray[np.float64_t,ndim=2] Q, int p, int q,
double eps):
    """
    Usage
    -----
    Compute all the free space on the boundary of cells in the diagram for polygonal chains P and Q and the given eps
    LF[(i,j)] is the free space of segments [Pi,Pi+1] from point  Qj
    BF[(i,j)] is the free space of segment [Qj,Qj+1] from point Pj

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance

    Returns
    -------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    """

    cdef dict LF,BF
    cdef int j,i
    cdef double Q_j0,Q_j1,Q_j10,Q_j11,P_i0,P_i1,P_i10,P_i11

    LF={}
    for j from 0 <= j < q:
        for i from 0 <= i < p-1 :
            Q_j0=Q[j,0]
            Q_j1=Q[j,1]
            P_i0=P[i,0]
            P_i1=P[i,1]
            P_i10=P[i+1,0]
            P_i11=P[i+1,1]

            LF.update({(i,j):c_free_line(Q_j0,Q_j1,eps,P_i0,P_i1,P_i10,P_i11)})
    BF={}
    for j from 0 <= j < q-1:
        for i from 0 <= i < p:
            Q_j0=Q[j,0]
            Q_j1=Q[j,1]
            Q_j10=Q[j+1,0]
            Q_j11=Q[j+1,1]
            P_i0=P[i,0]
            P_i1=P[i,1]

            BF.update({(i,j):c_free_line(P_i0,P_i1,eps,Q_j0,Q_j1,Q_j10,Q_j11)})
    return LF,BF

cdef c_compute_LR_BR(dict LF, dict BF,int p, int q):
    """
    Usage
    -----
    Compute all the free space,that are reachable from the origin (P[0,0],Q[0,0]) on the boundary of cells
    in the diagram for polygonal chains P and Q and the given free spaces LR and BR

    LR[(i,j)] is the free space, reachable from the origin, of segment [Pi,Pi+1] from point  Qj
    BR[(i,j)] is the free space, reachable from the origin, of segment [Qj,Qj+1] from point Pj

    Parameters
    ----------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q

    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    LR : dict, is the free space, reachable from the origin, of segments of P from points of Q
    BR : dict, is the free space, reachable from the origin, of segments of Q from points of P
    """

    cdef int i,j
    cdef dict LR,BR
    cdef bool rep

    if not(LF[(0,0)][0] <=0 and BF[(0,0)][0] <=0 and LF[(p-2,q-1),1]  >=1 and BF[(p-1,q-2)][1]  >=1 )  :
        rep = False
    else:
        LR = {(0,0):True}
        BR = {(0,0):True}
        for i from 1 <= i < p-1:

            if (LF[(i,0)][0]!=-1.0 or LF[(i,0)][1]!=-1) and (LF[(i-1,0)][0]==0 and LF[(i-1,0)][1]==1):
                LR[(i,0)]=True
            else:
                LR[(i,0)]=False
        for j from 1 <= j < q-1:
            if (BF[(0,j)][0]!=-1.0 or BF[(0,j)][1]!=-1.0) and (BF[(0,j-1)][0]==0 and BF[(0,j-1)][1]==1):
                BR[(0,j)]=True
            else:
                BR[(0,j)]=False
        for i from 0 <= i < p-1:
            for j from 0 <= j < q-1:
                if LR[(i,j)] or BR[(i,j)]:
                    if LF[(i,j+1)][0]!= -1.0 or LF[(i,j+1)][1]!=-1.0:
                        LR[(i,j+1)]=True
                    else:
                        LR[(i,j+1)]=False
                    if  BF[(i+1,j)][0]!=-1.0 or BF[(i+1,j)][1]!=-1.0:
                        BR[(i+1,j)]=True
                    else:
                        BR[(i+1,j)]=False
                else:
                    LR[(i,j+1)]=False
                    BR[(i+1,j)]=False
        rep = BR[(p-2,q-2)] or LR[(p-2,q-2)]
    return rep,LR,BR

def compute_LR_BR(dict LF, dict BF,int p, int q):
    """
    Usage
    -----
    Compute all the free space,that are reachable from the origin (P[0,0],Q[0,0]) on the boundary of cells
    in the diagram for polygonal chains P and Q and the given free spaces LR and BR

    LR[(i,j)] is the free space, reachable from the origin, of segment [Pi,Pi+1] from point  Qj
    BR[(i,j)] is the free space, reachable from the origin, of segment [Qj,Qj+1] from point Pj

    Parameters
    ----------
    LF : dict, free spaces of segments of P from points of Q
    BF : dict, free spaces of segments of Q from points of P
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q

    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    LR : dict, is the free space, reachable from the origin, of segments of P from points of Q
    BR : dict, is the free space, reachable from the origin, of segments of Q from points of P
    """

    cdef int i,j
    cdef dict LR,BR
    cdef bool rep

    if not(LF[(0,0)][0] <=0 and BF[(0,0)][0] <=0 and LF[(p-2,q-1)][1]  >=1 and BF[(p-1,q-2)][1]  >=1 )  :
        rep = False
    else:
        LR = {(0,0):True}
        BR = {(0,0):True}
        for i from 0 <= i < p-1:

            if (LF[(i,0)][0]!=-1.0 or LF[(i,0)][1]!=-1) and (LF[(i-1,0)][0]==0 and LF[(i-1,0)][1]==1):
                LR[(i,0)]=True
            else:
                LR[(i,0)]=False
        for j from 1 <= j < q-1:
            if (BF[(0,j)][0]!=-1.0 or BF[(0,j)][1]!=-1.0) and (BF[(0,j-1)][0]==0 and BF[(0,j-1)][1]==1):
                BR[(0,j)]=True
            else:
                BR[(0,j)]=False
        for i from 0 <= i < p-1:
            for j from 0 <= j < q-1:
                if LR[(i,j)] or BR[(i,j)]:
                    if LF[(i,j+1)][0]!= -1.0 or LF[(i,j+1)][1]!=-1.0:
                        LR[(i,j+1)]=True
                    else:
                        LR[(i,j+1)]=False
                    if  BF[(i+1,j)][0]!=-1.0 or BF[(i+1,j)][1]!=-1.0:
                        BR[(i+1,j)]=True
                    else:
                        BR[(i+1,j)]=False
                else:
                    LR[(i,j+1)]=False
                    BR[(i+1,j)]=False
        rep = BR[(p-2,q-2)] or LR[(p-2,q-2)]
    return rep,LR,BR


cdef bool c_decision_problem(np.ndarray[np.float64_t,ndim=2] P, np.ndarray[np.float64_t,ndim=2] Q,int p,int q, double
eps):
    """
    Usage
    -----
    Test is the frechet distance between trajectories P and Q are inferior to eps

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance

    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    """
    cdef dict LF,BF
    cdef bool rep

    LF,BF= c_compute_LF_BF(P,Q,p,q,eps)
    rep,_,_ =c_compute_LR_BR(LF,BF,p,q)
    return rep

def decision_problem(np.ndarray[np.float64_t,ndim=2] P, np.ndarray[np.float64_t,ndim=2] Q,int p,int q, double eps):
    """
    Usage
    -----
    Test is the frechet distance between trajectories P and Q are inferior to eps

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q
    param eps : float, reachability distance

    Returns
    -------
    rep : bool, return true if frechet distance is inf to eps
    """
    cdef dict LF,BF
    cdef bool rep

    LF,BF= c_compute_LF_BF(P,Q,p,q,eps)
    rep,_,_ =c_compute_LR_BR(LF,BF,p,q)
    return rep

cdef list c_compute_critical_values(np.ndarray[np.float64_t,ndim=2] P, np.ndarray[np.float64_t,ndim=2] Q,int p,int q):
    """
    Usage
    -----
    Compute all the critical values between trajectories P and Q

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q

    Returns
    -------
    cc : list, all critical values between trajectories P and Q
    """
    cdef double origin,end,end_point,Lij,Bij
    cdef set cc
    cdef int i,j

    origin = eucl_dist(P[0,0],P[0,1],Q[0,0],Q[0,1])
    end = eucl_dist(P[p-1,0],P[p-1,1],Q[q-1,0],Q[q-1,1])
    end_point=max(origin,end)
    cc=set([end_point])
    for i from 0 <= i < p-1:
        for j from 0 <= j < q-1:
            Lij=point_to_seg(Q[j,0],Q[j,1],P[i,0],P[i,1],P[i+1,0],P[i+1,1])
            if Lij>end_point:
                cc.add(Lij)
            Bij=point_to_seg(P[i,0],P[i,1],Q[j,0],Q[j,1],Q[j+1,0],Q[j+1,1])
            if Bij>end_point:
                cc.add(Bij)
    return sorted(list(cc))

def compute_critical_values(np.ndarray[np.float64_t,ndim=2] P, np.ndarray[np.float64_t,ndim=2] Q,int p,int q):
    """
    Usage
    -----
    Compute all the critical values between trajectories P and Q

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q
    param p : float, number of points in Trajectory P
    param q : float, number of points in Trajectory Q

    Returns
    -------
    cc : list, all critical values between trajectories P and Q
    """

    cdef double origin,end,end_point,Lij,Bij
    cdef set cc
    cdef int i,j

    origin = eucl_dist(P[0,0],P[0,1],Q[0,0],Q[0,1])
    end = eucl_dist(P[p-1,0],P[p-1,1],Q[q-1,0],Q[q-1,1])
    end_point=max(origin,end)
    cc=set([end_point])
    for i from 0 <= i < p-1:
        for j from 0 <= j < q-1:
            Lij=point_to_seg(Q[j,0],Q[j,1],P[i,0],P[i,1],P[i+1,0],P[i+1,1])
            if Lij>end_point:
                cc.add(Lij)
            Bij=point_to_seg(P[i,0],P[i,1],Q[j,0],Q[j,1],Q[j+1,0],Q[j+1,1])
            if Bij>end_point:
                cc.add(Bij)
    return sorted(list(cc))

def frechet(np.ndarray[np.float64_t,ndim=2] P,np.ndarray[np.float64_t,ndim=2] Q):
    """
    Usage
    -----
    Compute the frechet distance between trajectories P and Q

    Parameters
    ----------
    param P : px2 numpy_array, Trajectory P
    param Q : qx2 numpy_array, Trajectory Q

    Returns
    -------
    frech : float, the frechet distance between trajectories P and Q
    """

    cdef int p,q,m_i
    cdef double eps
    cdef bool rep
    cdef list cc
    p=len(P)
    q=len(Q)

    cc=c_compute_critical_values(P,Q,p,q)
    while(len(cc)!=1):
        m_i=len(cc)/2-1
        eps = cc[m_i]
        rep = c_decision_problem(P,Q,p,q,eps)
        if rep:
            cc=cc[:m_i+1]
        else:
            cc=cc[m_i+1:]
    return eps



