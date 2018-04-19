# distutils: language = c++

"""This module contains a Cython extension type which facilitates B-spline
interpolation of N-dimensional curves in N-space, making use of the
Bspline-Fortran library, which is available at
    https://github.com/jacobwilliams/bspline-fortran

Extension type defined here:
    NDCurveBsplineInterpolator

Written by Arne Magnus T. Løken as part of my master's thesis work in physics
at NTNU, spring 2018.

"""

cimport numpy as np
import numpy as np

cimport cython

from libcpp.vector cimport vector

ctypedef vector[double] dbl_vec

ctypedef vector[dbl_vec] dbl_vec_vec

from libc.math cimport pow as c_pow, sqrt as c_sqrt


cdef extern from "linkto.h":
    cppclass ItpCont:
        ItpCont() except +
        void init_interp(double *x, double *f,
                int kx, int nx, int ext) except +
        double eval_interp(double x, int dx) except +


ctypedef vector[ItpCont*] ItpCont_ctr

cdef class NDCurveBSplineInterpolator:
    """A Cython extension type which enables B-spline interpolation of
    N-dimensional curves in N-space, parametrized by a set of points.
    Makes use of the Bspline-Fortran library, which is available at
       https://github.com/jacobwilliams/bspline-fortran

    Methods defined here:
    NDCurveBSplineInterpolator.__init__(x,k,wraparound,pad_points,extrap)
    NDCurveBSplineInterpolator.__call__(s)
    """
    cdef:
        ItpCont_ctr c_classes
        readonly double[::1] s
        readonly double l
        double[::1] ret
        unsigned int ndim
        bint extrap

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def __init__(self, double[:,::1] x, int k = 4, bint wraparound = False,
            unsigned int pad_points = 0, bint extrap = False):
        """NDCurveBSplineInterpolator(x,k,wraparound,pad_points,extrap)

        Constructor for a B-spline interpolation of an N-dimensional curve,
        where each curve coordinate is parametrized as a function of
        a normalized pseudo-arclength parameter 0 <= s <= 1.

        param: x -- (C-contiguous) NumPy array of shape (n_points, n_dim),
                    containing the (Cartesian) coordinates of the pointwise
                    parametrized curve.
        OPTIONAL:
        param: k          -- Interpolation order (along all abscissae), as an
                             integer. Must satisfy 2 <= k <= n_points.
                             Default: k = 4.
        param: wraparound -- Boolean flag as to whether or not the parametrized
                             curve should be configured such that s = 0 and
                             s = 1 correspond to the same point in N-space.
                             Default: wraparound = False.
        param: pad_points -- Integer number of points to pad to either side
                             of the list of points, in order to facilitate
                             a smoother joint. Only really useful when
                             wraparound is True, i.e., when emulating periodic
                             boundary conditions. Default: pad_points = 0.
        param: extrap     -- Boolean flag as to whether or not the interpolator
                             should perform extrapolation when evaluated for
                             0 < s or s > 1. Default: extrap = False.
                             Note that extrapolation is generally not advised,
                             particularly for high dimensionality and/or
                             a large number of sampling points, as the
                             B-spline may not yield sensible results.

        The parameter s is readily available as a Python readonly memoryview.
        It can be extracted e.g. by means of a list comprehension, such as

            'npos, ndim = 40, 2
             phi = np.linspace(0,2*np.pi,npos,endpoint=False)
             r = 0.5 + np.cos(phi)
             x = np.empty((npos,ndim))
             x[...,0], x[...,1] = r*np.cos(phi), r*np.sin(phi)
             interp = NDCurveBSplineInterpolator(x,k=3,wraparound=True)
             s_external = [s for s in interp.s]
             (...)
            '

        """
        cdef:
            unsigned int npts
            int ext
            dbl_vec x_
            dbl_vec s_internal
            dbl_vec_vec pos_
            double diff2
            unsigned int i, j
        self.ndim = x.shape[1]
        self.extrap = extrap
        npts = x.shape[0]
        self.c_classes = ItpCont_ctr(self.ndim)
        self.ret = np.empty(self.ndim)
        self.s = np.empty(npts)

        if k < 2 or k > npts + 2*pad_points:
            raise RuntimeError('Impossible choice of interpolator order. See constructor docstring for details.')

        for i in range(self.ndim):
            self.c_classes[i] = new ItpCont()

        if wraparound:
            x_ = dbl_vec(npts+2*pad_points+1)
            s_internal = dbl_vec(npts+2*pad_points+1)
        else:
            x_ = dbl_vec(npts+2*pad_points)
            s_internal = dbl_vec(npts + 2*pad_points)
        pos_ = dbl_vec_vec(self.ndim, x_)

        for j in range(self.ndim):
            for i in range(pad_points):
                pos_[j][pad_points-1-i] = x[npts-1-i,j]

        for j in range(self.ndim):
            for i in range(npts):
                pos_[j][i+pad_points] = x[i,j]

        if wraparound:
            for j in range(self.ndim):
                pos_[j][npts+pad_points] = x[0,j]
            for j in range(self.ndim):
                for i in range(1,pad_points+1):
                    pos_[j][npts+pad_points+i] = x[i,j]
        else:
            for j in range(self.ndim):
                for i in range(pad_points):
                    pos_[j][npts+pad_points+i] = x[i,j]

        s_internal[pad_points] = 0
        for j in range(1,npts):
            diff2 = 0
            for i in range(self.ndim):
                diff2 += c_pow(pos_[i][j]-pos_[i][j-1],2)
            s_internal[pad_points+j] = s_internal[pad_points+j-1] + c_sqrt(diff2)

        if wraparound:
            diff2 = 0
            for i in range(self.ndim):
                diff2 += c_pow(pos_[i][npts]-pos_[i][npts-1],2)
            s_internal[npts+pad_points] = s_internal[npts+pad_points-1] + c_sqrt(diff2)

        if s_internal[s_internal.size()-1-pad_points] == 0:
            raise RuntimeError('Spline interpolation of multiple instances of a single point leads to undefined behaviour. Attempt aborted.')
        self.l = s_internal[s_internal.size()-pad_points-1]
        for j in range(s_internal.size() - 2*pad_points):
            s_internal[j+pad_points] /= s_internal[s_internal.size()-pad_points-1]


        if wraparound:
            for j in range(pad_points):
                s_internal[pad_points-1-j] = s_internal[npts+pad_points-j-1] - 1
                s_internal[s_internal.size() - 1 - j] = 1 + s_internal[2*pad_points-j]#s_internal[pad_points + j + 1]
        else:
            for j in range(pad_points):
                s_internal[pad_points-1-j] = s_internal[npts+pad_points-j-2] - 1
                s_internal[s_internal.size() - 1 - j] = 1 + s_internal[2*pad_points-j]#s_internal[pad_points + j + 1]

        if self.extrap:
            ext = 1
        else:
            ext = 0

        if wraparound:
            for i in range(self.ndim):
                self.c_classes[i].init_interp(&s_internal[0],&pos_[i][0],k,npts+1+2*pad_points,ext)
        else:
            for i in range(self.ndim):
                self.c_classes[i].init_interp(&s_internal[0],&pos_[i][0],k,npts+2*pad_points,ext)
        for i in range(npts):
            self.s[i] = s_internal[pad_points+i]


    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double s):
        """NDCurveBSplineInterpolator.__call__(s)

        Evaluates the N-dimensional B-spline interpolant at the parameter
        value 's'. If extrapolation is not allowed (cf. constructor),
        evaluations for 0 < s or s > 1 will return zero.

        param: s -- Parameter at which to evaluate the B-spline interpolant.

        return: vec -- (C-contiguous) NumPy array of shape (n_dim), containing
                       the interpolated coordinates.

        """
        cdef:
            unsigned int i

        if not self.extrap and (s < 0 or s > 1):
            return np.zeros(3)
        for i in range(self.ndim):
            self.ret[i] = self.c_classes[i].eval_interp(s,0)
        return np.array(self.ret,copy=True)
#        return np.copy(self.ret)

    def __dealoc__(self):
        cdef:
            unsigned int i

        for i in range(self.ndim):
            if self.c_classes[i] is not NULL:
                del self.c_classes[i]
