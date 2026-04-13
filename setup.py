from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name="src.ising_core",  
        sources=["src/ising_core.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/O2']
    ),
    Extension(
        name = "src.ising_core_p",
        sources = ["src/ising_core_p.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['/pO2', '/openmp'],
        extra_link_args=['/openmp']
    )
]
setup(
    ext_modules=cythonize(extensions)
)