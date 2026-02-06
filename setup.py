# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="src.ising_core",  
        sources=["src/ising_core.pyx"],
    )
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()]
)