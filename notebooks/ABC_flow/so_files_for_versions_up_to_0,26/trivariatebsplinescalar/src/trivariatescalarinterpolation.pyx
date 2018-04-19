# distutils: language = c++

"""This module contains a Cython extension type which facilitates B-spline
interpolation of trivariate scalar fields in R^3, making use of the
Bspline-Fortran library, which is available at
    https://github.com/jacobwilliams/bspline-fortran

Extension types defined here:
    TrivariateSpline

Written by Arne Magnus T. Løken as part of my master's thesis work in physics
at NTNU, spring 2018.

"""
cimport numpy as np
import numpy as np

cimport cython

from libcpp.vector cimport vector

cdef extern from "linkto.h":
    cppclass ItpCont:
        ItpCont() except +
        void init_interp(double *x, double *y, double *z, double *f,
                int kx, int ky, int kz, int nx, int ny, int nz, int ext) except +
        double eval_interp(double x, double y, double z, int dx, int dy, int dz) except +
        void kill_interp() except +

ctypedef vector[double] dbl_vec
ctypedef vector[dbl_vec] dbl_mat

cdef class TrivariateSpline:
    """A Cython extension type tailor-made for computing a special-purpose
    (higher order) spline interpolation of a trivariate scalar field.

    Methods defined here:
    TrivariateSpline.__init__(x,y,z,f,kx,ky,kz)
    TrivariateSpline.__call__(pos)
    TrivariateSpline.grad(pos)
    TrivariateSpline.hess(pos)

    Version: 0.2

    """
    cdef:
        ItpCont *interp
        double _grad_[3]
        double _hess_[3][3]
        double[::1] grad_
        double[:,::1] hess_

    def __cinit__(self):
        self.grad_ = self._grad_
        self.hess_ = self._hess_
        self.interp = new ItpCont()

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:,:,::1] f,
            int kx = 3, int ky = 3, int kz = 3, bint extrap = False):
        """TrivariateSpline.__init__(x,y,z,xi,kx,ky,kz)

        Constructor for the three-dimensional, trivariate B-spline interpolator
        extension type.

        param: x  -- Sampling points along the x abscissa, as a (C-contiguous)
                     NumPy array of shape (nx) and type np.float64
        param: y  -- Sampling points along the y abscissa, as a (C-contiguous)
                     NumPy array of shape (ny) and type np.float64
        param: z  -- Sampling points along the z abscissa, as a (C-contiguous)
                     NumPy array of shape (nz) and type np.float64
        param: f  -- Sampled scalar field at the grid spanned by the input
                     arrays x, y and z, as a (C-contiguous) NumPy array of
                     shape (nx,ny,nz) and type np.float64
        OPTIONAL:
        param: kx --     Interpolation order along the x-axis. Must satisfy
                         2 <= kx <= nx. Default: kx = 3.
        param: ky --     Interpolation order along the y-axis. Must satisfy
                         2 <= ky <= ny. Default: ky = 3.
        param: kz --     Interpolation order along the z-axis. Must satisfy
                         2 <= kz <= nz. Default: kz = 3.
        param: extrap -- Boolean flag indicating whether or not extrapolation
                         is allowed. Generally not advised, particularly for
                         high order interpolation, as the results tend to
                         become noisy outside of the sampling domain.

        """
        cdef:
            dbl_vec f_flat
            int nx = x.shape[0], ny = y.shape[0], nz = z.shape[0]
            int i, j, k
            int ext

        if x.shape[0] != f.shape[0] or y.shape[0] != f.shape[1] or z.shape[0] != f.shape[2]:
            raise RuntimeError('Array dimensions not aligned!')

        if (kx < 2 or kx > x.shape[0]) or (ky < 2 or ky > y.shape[0]) or (kz < 2 or kz > z.shape[0]):
            raise RuntimeError('Invalid interpolator order choice!')

        f_flat = dbl_vec(nx*ny*nz)
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    f_flat[i+nx*(j+k*ny)] = f[i,j,k]

        if extrap:
            ext = 1
        else:
            ext = 0

        self.interp.init_interp(&x[0],&y[0],&z[0],&f_flat[0],kx,ky,kz,nx,ny,nz,ext)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double[::1] pos):
        """TrivariateSpline.__call__(pos)

        Returns a B-spline interpolation of the scalar field at 'pos'.

        If 'pos' is outside of the sampling domain and extrapolation is not
        allowed, this routine returns zero.

        param: pos -- Three-component (C-contiguous) NumPy array,
                      containing the Cartesian coordinates at which a spline
                      interpolated vector is sought

        return: f  -- B-spline interpolated scalar at 'pos'.

        """
        if pos.shape[0] != 3:
            raise RuntimeError('This interpolation routine is custom-built for three-dimensional data.')

        return self.interp.eval_interp(pos[0],pos[1],pos[2],0,0,0)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def grad(self, double[::1] pos):
        """TrivariateSpline.grad(pos)

        Returns a B-spline interpolation of the gradient of the scalar field at
        'pos'.

        If 'pos' is outside of the sampling domain and extrapolation is not
        allowed, this routine returns a three-component NumPy array of zeros.

        param: pos -- Three-component (C-contiguous) NumPy array,
                      containing the Cartesian coordinates at which a spline
                      interpolated vector is sought

        return: grad -- B-spline interpolated gradient of the scalar field at
                        'pos'.

        """
        cdef:
            double[::1] grad_ = self.grad_
        if pos.shape[0] != 3:
            raise RuntimeError('This interpolation routine is custom-built for three-dimensional data.')

        grad_[0] = self.interp.eval_interp(pos[0],pos[1],pos[2],1,0,0)
        grad_[1] = self.interp.eval_interp(pos[0],pos[1],pos[2],0,1,0)
        grad_[2] = self.interp.eval_interp(pos[0],pos[1],pos[2],0,0,1)
        return np.copy(grad_)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def hess(self, double[::1] pos):
        """TrivariateSpline.hess(pos)

        Returns a B-spline interpolation of the Hessian of the scalar field at
        'pos'.

        The Hessian components are organized as follows:

        [[fxx, fxy, fxz],
         [fyx, fyy, fyz],
         [fzx, fzy, fzz]]

        If 'pos' is outside of the sampling domain and extrapolation is not
        allowed, this routine returns a three-by-three-component NumPy array of
        zeros.

        param: pos -- Three-component (C-contiguous) NumPy array,
                      containing the Cartesian coordinates at which a spline
                      interpolated vector is sought

        return: hess -- B-spline interpolated Hessian of the scalar field at
                        'pos'.

        """
        cdef:
            double[:,::1] hess_ = self.hess_
        if pos.shape[0] != 3:
            raise RuntimeError('This interpolation routine is custom-built for three-dimensional data.')

        hess_[0,0] = self.interp.eval_interp(pos[0],pos[1],pos[2],2,0,0)
        hess_[0,1] = self.interp.eval_interp(pos[0],pos[1],pos[2],1,1,0)
        hess_[0,2] = self.interp.eval_interp(pos[0],pos[1],pos[2],1,0,1)
        hess_[1,0] = self.interp.eval_interp(pos[0],pos[1],pos[2],1,1,0)
        hess_[1,1] = self.interp.eval_interp(pos[0],pos[1],pos[2],0,2,0)
        hess_[1,2] = self.interp.eval_interp(pos[0],pos[1],pos[2],0,1,1)
        hess_[2,0] = self.interp.eval_interp(pos[0],pos[1],pos[2],1,0,1)
        hess_[2,1] = self.interp.eval_interp(pos[0],pos[1],pos[2],0,1,1)
        hess_[2,2] = self.interp.eval_interp(pos[0],pos[1],pos[2],0,0,2)
        return np.copy(hess_)

    def __dealoc__(self):
        if self.interp is not NULL:
            del self.interp
