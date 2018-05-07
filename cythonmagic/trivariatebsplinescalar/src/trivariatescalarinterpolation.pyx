# distutils: language = c++

"""
This module contains a Cython extension type which facilitates B-spline
interpolation of of trivariate scalar fields in R^3, making use of
the Bspline-Fortran library, which is available at
    https://github.com/jacobwilliams/bspline-fortran

Extension types defined here:
    ScalarTrivariateSpline

Written by Arne Magnus T. Løken as part of my master's thesis work in physics
at NTNU, spring 2018.

"""
cimport numpy as np
import numpy as np

cimport cython

cdef extern from "linkto.h":
    cppclass ItpCont:
        ItpCont() except +
        void init_interp(double *x, double *y, double *z, double *f,
                int kx, int ky, int kz, int nx, int ny, int nz, int ext) except +
        double eval_interp(double x, double y, double z, int dx, int dy, int dz) except +
        void kill_interp() except +


# In-house wrapper class for the bspline-3d fortran class
@cython.internal
cdef class Interpolator:
    cdef:
        ItpCont *interp
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self, double[::1] x, double[::1] y, double[::1] z, double[::1] f,
            int kx, int ky, int kz, bint extrap):
        cdef:
            int ext
        self.interp = new ItpCont()
        if extrap:
            ext = 1
        else:
            ext = 0
        self.interp.init_interp(&x[0],&y[0],&z[0],&f[0],kx,ky,kz,x.shape[0],y.shape[0],z.shape[0],ext)
    cdef void _ev_(self, double x, double y, double z, int kx, int ky, int kz, double *ret):
        ret[0] = self.interp.eval_interp(x,y,z,kx,ky,kz)
    def __dealoc__(self):
        if self.interp is not NULL:
            del self.interp

cdef class ScalarTrivariateSpline:
    """
    A Cython extension type which facilitates simultaneous interpolation of
    time-dependent vector fields in R^3 along all thre spatial axes, as well
    as the time abscissa.

    Methods defined here:
    ScalarTrivariateSpline.__init__(x,y,z,f,kx,ky,kz,extrap)
    ScalarTrivariateSpline.__call__(pos)
    ScalarTrivariateSpline.grad(pos)
    ScalarTrivariateSpline.hess(pos)

    Version: 0.1
    """
    cdef:
        Interpolator itp
        double _grd_[3]
        double _hss_[3][3]
        double[::1] grd
        double[:,::1] hss
    def __cinit__(self):
        self.grd = self._grd_
        self.hss = self._hss_
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[::1] f,
            int kx = 4, int ky = 4, int kz = 4, bint extrap = False):
        """
        ScalarTrivariateSpline.__init__(x,y,z,f,kx,ky,kz,extrap)

        Constructor for a TrivectorQuadvariateSpline instance.

        Parameters
        ----------
        x : (nx,) array_like
            A (C-contiguous) (NumPy) array of type np.float64 and shape (nx,),
            containing the sampling points along the x abscissa.
            Must be strictly increasing.
        y : (ny,) array_like
            A (C-contiguous) (NumPy) array of type np.float64 and shape (ny,),
            containing the sampling points along the y abscissa.
            Must be strictly increasing.
        z : (nz,) array_like
            A (C-contiguous) (NumPy) array of type np.float64 and shape (nz,),
            containing the sampling points along the z, abscissa.
            Must be strictly increasing.
        f : (nx*ny*nz,) array_like
            A (C-contiguous) (NumPy) array of type np.float64 and shape
            (nx*ny*nz,), containing the sampled scalar values.

            The array must be sorted in Fortran order. That is,
            f[i+nx*(j+ny*k)] should contain the scalar sampled
            at (x[i],y[j],z[k]).
            This is most easily done by:
                1) Initializing an array of shape (nx,ny,nz),
                   filling it with the corresponding scalar values
                2) Flattening the array in Fortran order before
                   assigning to f, i.e.,
                   f = f_tmp.ravel(order='F')
        kx: integer, optional
            Interpolation order (= degree of polynomial pieces + 1) along the
            x abscissa. Default: kx = 4.
            Must satisfy 2 < kx < nx
        ky: integer, optional
            Interpolation order (= degree of polynomial pieces + 1) along the
            y abscissa. Default: ky = 4.
            Must satisfy 2 < ky < ny
        kz: integer, optional
            Interpolation order (= degree of polynomial pieces + 1) along the
            z abscissa. Default: kz = 4.
            Must satisfy 2 < kz < nz
        extrap : bool, optional
            Flag indicating whether or not extrapolation outside of the
            original sampling domain is allowed. If extrap is false and
            one attempts to evaluate the interpolated vector field outside of
            the sampling domain, zero is returned.
            Extrapolation is generally not advised for high interpolation
            orders.
            Default: extrap = False.

        """
        cdef:
            int nx = x.shape[0], ny = y.shape[0], nz = z.shape[0]

        if nx*ny*nz != f.shape[0]:
            raise ValueError('Array dimensions not aligned!')

        if (kx < 2 or kx > nx) or (ky < 2 or ky > ny) or (kz < 2 or kz > nz):
            raise ValueError('Invalid interpolator order choice!')

        self.itp = Interpolator(x, y, z, f, kx, ky, kz, extrap)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, double[::1] pos):
        """
        ScalarTrivariateSpline.__call__(pos)

        Evaluates the interpolated vector field at at the coordinates specified
        by pos.

        Parameters
        ----------
        pos: (3,) ndarray
          A (C-contiguous) (NumPy) array containing the (Cartesian) coordinates
          of the point in R^3 at which the vector field is to be interpolated

        Returns
        -------
        val : double
          Interpolated scalar.
          If extrap = False and an interpolation is requested outside of the
          sampling domain, val = 0.

        """
        cdef:
            double tmp
        self.itp._ev_(pos[0],pos[1],pos[2],0,0,0,&tmp)
        return tmp
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def grad(self, double[::1] pos):
        """
        ScalarTrivariateSpline.grad(pos)

        Evaluates the gradient of the interpolated scalar field at the
        coordinates specified by pos.

        Parameters
        ----------
        pos: (3,) ndarray
          A (C-contiguous) (NumPy) array containing the (Cartesian) coordinates
          of the point in R^3 at which the vector field is to be interpolated

        Returns
        -------
        grad : (3,) ndarray
          (Interpolated) gradient.
          If extrap = False and an interpolation is requested outside of the
          sampling domain, grad = np.zeros(3).

        """
        cdef:
            double[::1] grd = self.grd
            double tmp
        self.itp._ev_(pos[0],pos[1],pos[2],1,0,0,&tmp)
        grd[0] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],0,1,0,&tmp)
        grd[1] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],0,0,1,&tmp)
        grd[2] = tmp
        return np.copy(grd)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def hess(self, double[::1] pos):
        """
        ScalarTrivariateSpline.hess(pos)

        Evaluates the Hessian matrix of the interpolated scalar field at the
        coordinates specified by pos.

        Parameters
        ----------
        pos: (3,) ndarray
          A (C-contiguous) (NumPy) array containing the (Cartesian) coordinates
          of the point in R^3 at which the vector field is to be interpolated

        Returns
        -------
        hess : (3,3) ndarray
          (Interpolated) Hessian matrix.
          If extrap = False and an interpolation is requested outside of the
          sampling domain, hess = np.zeros(3,3).

        """
        cdef:
            double[:,::1] hss = self.hss
            double tmp
        self.itp._ev_(pos[0],pos[1],pos[2],2,0,0,&tmp)
        hss[0,0] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],1,1,0,&tmp)
        hss[0,1] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],1,0,1,&tmp)
        hss[0,2] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],1,1,0,&tmp)
        hss[1,0] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],0,2,0,&tmp)
        hss[1,1] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],0,1,1,&tmp)
        hss[1,2] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],1,0,1,&tmp)
        hss[2,0] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],0,1,1,&tmp)
        hss[2,1] = tmp
        self.itp._ev_(pos[0],pos[1],pos[2],0,0,2,&tmp)
        hss[2,2] = tmp
        return np.copy(hss)

## The following is a deprecated implementation of a B-spline interpolation
## object for three-dimensional vector fields, designed with advection
## based on model data. Its replacement, TrivectorQuadvariateSpline, enables
## interpolation in time as well as along the spatial abscissa.
##
## Regardless, the implementation is kept for future reference.
#cdef class VectorTrivariateSpline:
#    cdef:
#        Interpolator itpx, itpy, itpz
#        double _vct_[3]
#        double[::1] vct
#        double _jcb_[3][3]
#        double[:,::1] jcb
#    def __cinit__(self):
#        self.vct = self._vct_
#        self.jcb = self._jcb_
#    def __init__(self, double[::1] x, double[::1] y, double[::1] z, double[:,::1] f,
#            int kx = 4, int ky = 4, int kz = 4, bint extrap = False):
#        cdef:
#            int nx = x.shape[0], ny = y.shape[0], nz = z.shape[0]
#
#        if f.shape[0] != 3 or nx*ny*nz != f.shape[1]:
#            raise ValueError('Array dimensions not aligned!')
#
#        if (kx < 2 or kx > x.shape[0]) or (ky < 2 or ky > y.shape[0]) or (kz < 2 or kz > z.shape[0]):
#            raise ValueError('Invalid interpolator order choice!')
#
#        self.itpx = Interpolator(x, y, z, f[0], kx, ky, kz, extrap)
#        self.itpy = Interpolator(x, y, z, f[1], kx, ky, kz, extrap)
#        self.itpz = Interpolator(x, y, z, f[2], kx, ky, kz, extrap)
#
#    def __call__(self, double[::1] pos):
#        cdef:
#            double[::1] vct = self.vct
#        vct[0] = self.itpx._ev_(pos[0],pos[1],pos[2],0,0,0)
#        vct[1] = self.itpy._ev_(pos[0],pos[1],pos[2],0,0,0)
#        vct[2] = self.itpz._ev_(pos[0],pos[1],pos[2],0,0,0)
#        return np.copy(vct)
#    def jac(self, double[::1] pos):
#        cdef:
#            double[:,::1] jac = self.jac
#        jac[0,0] = self.itpx._ev_(pos[0],pos[1],pos[2],1,0,0)
#        jac[0,1] = self.itpx._ev_(pos[0],pos[1],pos[2],0,1,0)
#        jac[0,2] = self.itpx._ev_(pos[0],pos[1],pos[2],0,0,1)
#        jac[1,0] = self.itpy._ev_(pos[0],pos[1],pos[2],1,0,0)
#        jac[1,1] = self.itpy._ev_(pos[0],pos[1],pos[2],0,1,0)
#        jac[1,2] = self.itpy._ev_(pos[0],pos[1],pos[2],0,0,1)
#        jac[2,0] = self.itpz._ev_(pos[0],pos[1],pos[2],1,0,0)
#        jac[2,1] = self.itpz._ev_(pos[0],pos[1],pos[2],0,1,0)
#        jac[2,2] = self.itpz._ev_(pos[0],pos[1],pos[2],0,0,1)
#        return np.copy(jac)
