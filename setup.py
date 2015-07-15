from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("traj_dist.c_basic_geographical", [ "traj_dist/c_basic_geographical.pyx" ]),
               Extension("traj_dist.c_basic_euclidean", [ "traj_dist/c_basic_euclidean.pyx" ]),
               Extension("traj_dist.c_sspd", [ "traj_dist/c_sspd.pyx" ]),
               Extension("traj_dist.c_dtw", [ "traj_dist/c_dtw.pyx" ]),
               Extension("traj_dist.c_lcss", [ "traj_dist/c_lcss.pyx" ]),
               Extension("traj_dist.c_hausdorff", [ "traj_dist/c_hausdorff.pyx" ]),
               Extension("traj_dist.c_discret_frechet", [ "traj_dist/c_discret_frechet.pyx" ]),
               Extension("traj_dist.c_frechet", [ "traj_dist/c_frechet.pyx" ]),
               Extension("traj_dist.c_distance", [ "traj_dist/c_distance.pyx" ]),
               Extension("traj_dist.c_segment_distance", [ "traj_dist/c_segment_distance.pyx" ])]

setup(
    name = "trajectory_distance",
    version = "1.0",
    author = "Brendan Guillouet",
    author_email = "brendan.guillouet@gmail.com",
    cmdclass = { 'build_ext': build_ext },
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
    description = "Distance to compare trajectories in Cython",
    packages = ['traj_dist',]
)
