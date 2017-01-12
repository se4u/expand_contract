from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "fast_count",
        ["fast_count.pyx"],
        include_dirs = [np.get_include()]),
]
setup(
    name = "fast_count",
    ext_modules = cythonize(extensions),
)
