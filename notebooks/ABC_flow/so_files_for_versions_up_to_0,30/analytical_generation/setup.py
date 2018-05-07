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
        Extension(name = 'analytical_field',
                    sources = ['./src/analytical_field.pyx'],
                    language = 'c++',
                  include_dirs = [numpy.get_include()],
                ),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension('analytical_field', ['./build/tmp/analytical_field.cpp'],
                  include_dirs = [numpy.get_include()]),
    ]

setup(
    name = 'analytical_field',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
)

