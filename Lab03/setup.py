from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    include_dirs=[np.get_include()],
    name="convolve",
    ext_modules=cythonize("convolve.pyx"),
    zip_safe=False,
)