# trajectory_distance
=====================

**trajectory_distance** is a Python module for computing distance between trajectory objects.
It is implemented in both Python and Cython.

## Description

**trajectory_distance** contains 9 distances between trajectory.

1. SSPD (Symmetric Segment-Path Distance)
2. OWD  (One-Way Distance)
3. Hausdorff
4. Frechet
5. Discret Frechet
6. DTW (Dynamic Time Warping)
7. LCSS (Longuest Common Subsequence)
8. ERP (Edit distance with Real Penalty)
9. EDR (Edit Distance on Real sequence)

and one distance between segments (10).

These distance are implemented in Cython for 2-D Trajectory. And Python for all dimensions. 


 

## Dependencies

trajectory_distance is tested to work under Python 2.7.

The required dependencies to build the software are NumPy >= 1.9.1, Cython >= 0.21.2, shapely >= 1.5.6, Geohash and a working C/C++ compiler.

## Install

This package uses distutils. To install in your home directory, use:

```
python setup.py install 
```

## How to use it

## References

1.  *P.  Besse,  B.  Guillouet,  J.-M.  Loubes,  and  R.  Francois,  “Review  and perspective   for   distance based trajectory clustering,”
arXiv preprint arXiv:1508.04904, 2015.*
2. *B. Lin and J. Su, “Shapes based trajectory queries for moving objects,”
in
Proceedings  of  the  13th  annual  ACM  international  workshop  on
Geographic information systems
.    ACM, 2005, pp. 21–30.*
3. *F. Hausdorff, “Grundz uge der mengenlehre,” 1914*
4. *M. M. Fr
echet, “Sur quelques points du calcul fonctionnel,”
Rendiconti
del  Circolo  Matematico  di  Palermo  (1884-1940)
,  vol.  22,  no.  1,  pp.
1–72, 1906.*
5. *H.  Alt  and  M.  Godau,  “Computing  the  frechet  distance  between  two
polygonal curves,”
International Journal of Computational Geometry &
Applications
, vol. 5, no. 01n02, pp. 75–91, 1995.*
6. *D. J. Berndt and J. Clifford , “Using dynamic time warping to find patterns in time series.” in KDD workshop, vol. 10, no. 16. Seattle, WA, 1994, pp. 359–370* 
7. *M. Vlachos, G. Kollios, and D. Gunopulos, “Discovering similar multi-
dimensional trajectories,” in
Data Engineering, 2002. Proceedings. 18th
International Conference on
.IEEE, 2002, pp. 673–684*
8. *L.  Chen  and  R.  Ng,  “On  the  marriage  of  lp-norms  and  edit  distance,”
in
Proceedings  of  the  Thirtieth  international  conference  on  Very  large
data bases-Volume 30
.    VLDB Endowment, 2004, pp. 792–803.*
9. *L. Chen, M. T.
 ̈
Ozsu, and V. Oria, “Robust and fast similarity search for
moving object trajectories,” in
Proceedings of the 2005 ACM SIGMOD
international  conference  on  Management  of  data
.      ACM,  2005,  pp.
491–502.*
10. *J.-G. Lee, J. Han, and K.-Y. Whang, “Trajectory clustering: a partition-
and-group  framework,”  in
Proceedings  of  the  2007  ACM  SIGMOD
international  conference  on  Management  of  data
.      ACM,  2007,  pp.
593–604.*
