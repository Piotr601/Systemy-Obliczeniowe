from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    include_dirs=[np.get_include()],
    name="Z3x",
    ext_modules=cythonize("Z3x.pyx"),
    zip_safe=False,
)