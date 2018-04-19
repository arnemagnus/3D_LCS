from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

if use_cython:
    ext_modules += [
        Extension(name = 'trivariatevectorlinearinterpolation',
                    sources = ['trivariatevectorlinearinterpolation.pyx'],
                    language = 'c++',
                  include_dirs = [numpy.get_include()],
                ),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension('trivariatevectorlinearinterpolation', ['trivariatevectorlinearinterpolation.cpp'],
                  include_dirs = [numpy.get_include()]),
    ]

setup(
    name = 'trivariatevectorlinearinterpolation',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
)

