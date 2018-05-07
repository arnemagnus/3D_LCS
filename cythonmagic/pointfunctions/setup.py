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
        Extension(name = 'auxiliarypointstuff',
                    sources = ['src/auxiliarypointstuff.pyx'],
                    language = 'c++',
                  include_dirs = [numpy.get_include()],
                ),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension('auxiliarypointstuff', ['src/auxiliarypointstuff.cpp'],
                  include_dirs = [numpy.get_include()]),
    ]

setup(
    name = 'auxiliarypointstuff',
    cmdclass = cmdclass,
    ext_modules = ext_modules,
)

