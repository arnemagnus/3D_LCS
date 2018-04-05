from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(
        ext_modules = cythonize(Extension(
            "cython_numerical_integrators",
            sources = ["cython_numerical_integrators.pyx"],
            language = "c++",
            extra_compile_args = ["-fopenmp"],
            extra_link_args = ["-fopenmp"]
            )))
