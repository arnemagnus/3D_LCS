# distutils: language = c++

"""
This module contains a Cython extension type which facilitates B-spline
interpolation of quadvariate time-dependent vector fields in R^3, making use of
the Bspline-Fortran library, which is available at
    https://github.com/jacobwilliams/bspline-fortran

Extension types defined here:
    TrivectorQuadvariateSpline

Written by Arne Magnus T. Løken as part of my master's thesis work in physics
at NTNU, spring 2018.

"""
cimport numpy as np
import numpy as np

cimport cython

cdef extern from "linkto.h":
    cppclass ItpCont:
        ItpCont() except +
        void init_interp(double *x, double *y, double *z, double *q, double *f,
                int kx, int ky, int kz, int kq, int nx, int ny, int nz, int nq, int ext) except +
        double eval_interp(double x, double y, double z, double q, int dx, int dy, int dz, int dq) except +
        void kill_interp() except +


# In-house wrapper class for the bspline-4d fortran class
@cython.internal
cdef class Interpolator:
    cdef:
        ItpCont *interp
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __cinit__(self, double[::1] x, double[::1] y, double[::1] z, double[::1] q, double[::1] f,
            int kx, int ky, int kz, int kq, bint extrap):
        cdef:
            int ext
        self.interp = new ItpCont()
        if extrap:
            ext = 1
        else:
            ext = 0
        self.interp.init_interp(&x[0],&y[0],&z[0],&q[0],&f[0],kx,ky,kz,kq,x.shape[0],y.shape[0],z.shape[0],q.shape[0],ext)
    cdef void _ev_(self, double x, double y, double z, double q, int kx, int ky, int kz, int kq, double*ret):
        ret[0] = self.interp.eval_interp(x,y,z,q,kx,ky,kz,kq)
    def __dealoc__(self):
        if self.interp is not NULL:
            del self.interp


cdef class TrivectorQuadvariateSpline:
    """
    A Cython extension type which facilitates simultaneous interpolation of
    time-dependent vector fields in R^3 along all thre spatial axes, as well
    as the time abscissa.

    Methods defined here:
        TrivectorQuadvariateSpline.__init__(t,x,y,z,v,kt,kx,ky,kz,extrap)
        TrivectorQuadvariateSpline.__call__(t,pos)
        TrivectorQuadvariateSpline.jac(t,pos)

    Version: 0.1
    """
    cdef:
        Interpolator itpx, itpy, itpz
        double _vct_[3]
        double _jcb_[3][3]
        double[::1] vct
        double[:,::1] jcb
    def __cinit__(self):
        self.vct = self._vct_
        self.jcb = self._jcb_
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] t, double[::1] x, double[::1] y, double[::1] z, double[:,::1] v,
            int kt = 4, int kx = 4, int ky = 4, int kz = 4, bint extrap = False):
        """
        TrivectorQuadvariateSpline.__init__(t,x,y,z,v,kt,kx,ky,kz,extrap)

        Constructor for a TrivectorQuadvariateSpline instance.

        Parameters
        ----------
        t : (nt,) array_like
            A (C-contiguous) (NumPy) array of type np.float64 and shape (nt,),
            containing the sampling points along the time abscissa.
            Must be strictly increasing.
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
        v : (3,nt*nx*ny*nz) array_like
            A (C-contiguous) (NumPy) array of type np.float64 and shape
            (3,nt*nx*ny*nz), containing the sampled vector values.
            v[0] should contain the x-component, v[1] the y-component and
            v[2] the z-component.

            Along axis 1, the array must be sorted in Fortran order. That is,
            v[:,i+nt*(j+nx*(k+ny*m))] should contain the vector sampled
            at (t[i],x[j],y[k],z[m]).
            This is most easily done by:
                1) Initializing an array of shape (nt,nx,ny,nz) for each
                   vector component, filling it with the corresponding component
                   values
                2) Flattening each respective array in Fortran order before
                   assigning to v, i.e.,
                   v[0] = u1_tmp.ravel(order='F')
                   v[1] = u2_tmp.ravel(order='F')
                   v[2] = u3_tmp.ravel(order='F')
        kt: integer, optional
            Interpolation order (= degree of polynomial pieces + 1) along the
            t abscissa. Default: kt = 4.
            Must satisfy 2 < kt < nt
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
            int nt = t.shape[0], nx = x.shape[0], ny = y.shape[0], nz = z.shape[0]

        if v.shape[0] != 3 or nt*nx*ny*nz != v.shape[1]:
            raise ValueError('Array dimensions not aligned!')

        if (kt < 2 or kt > nt) or (kx < 2 or kx > nx) or (ky < 2 or ky > ny) or (kz < 2 or kz > nz):
            raise ValueError('Invalid interpolator order choice!')

        self.itpx = Interpolator(t,x,y,z,v[0],kt,kx,ky,kz,extrap)
        self.itpy = Interpolator(t,x,y,z,v[1],kt,kx,ky,kz,extrap)
        self.itpz = Interpolator(t,x,y,z,v[2],kt,kx,ky,kz,extrap)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos):
        """
        TrivectorQuadvariateSpline.__call__(t,pos)

        Evaluates the interpolated vector field at time t, at the coordinates
        specified by pos.

        Parameters
        ----------
        t : double
          The time at which to interpolate the vector field
        pos: (3,) ndarray
          A (C-contiguous) (NumPy) array containing the (Cartesian) coordinates
          of the point in R^3 at which the vector field is to be interpolated

        Returns
        -------
        vct : (3,) ndarray
          Interpolated vector.
          If extrap = False and an interpolation is requested outside of the
          sampling domain, vct = np.zeros(3)

        """
        cdef:
            double tmp
            double[::1] vct = self.vct
        self.itpx._ev_(t,pos[0],pos[1],pos[2],0,0,0,0,&tmp)
        vct[0] = tmp
        self.itpy._ev_(t,pos[0],pos[1],pos[2],0,0,0,0,&tmp)
        vct[1] = tmp
        self.itpz._ev_(t,pos[0],pos[1],pos[2],0,0,0,0,&tmp)
        vct[2] = tmp
        return np.copy(vct)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def jac(self, double t, double[::1] pos):
        """
        TrivectorQuadvariateSpline.jac(t,pos)

        Evaluates the Jacobian matrix of the interpolated vector field at time
        t, at the coordinates specified by pos.

        Parameters
        ----------
        t : double
          The time at which to interpolate the vector field
        pos: (3,) ndarray
          A (C-contiguous) (NumPy) array containing the (Cartesian) coordinates
          of the point in R^3 at which the vector field is to be interpolated

        Returns
        -------
        jac : (3,3) ndarray
          (Interpolated) Jacobian matrix.
          If extrap = False and an interpolation is requested outside of the
          sampling domain, jac = np.zeros((3,3))

        """
        cdef:
            double tmp
            double[:,::1] jcb = self.jcb
        self.itpx._ev_(t,pos[0],pos[1],pos[2],0,1,0,0,&tmp)
        jcb[0,0] = tmp
        self.itpx._ev_(t,pos[0],pos[1],pos[2],0,0,1,0,&tmp)
        jcb[0,1] = tmp
        self.itpx._ev_(t,pos[0],pos[1],pos[2],0,0,0,1,&tmp)
        jcb[0,2] = tmp

        self.itpy._ev_(t,pos[0],pos[1],pos[2],0,1,0,0,&tmp)
        jcb[1,0] = tmp
        self.itpy._ev_(t,pos[0],pos[1],pos[2],0,0,1,0,&tmp)
        jcb[1,1] = tmp
        self.itpy._ev_(t,pos[0],pos[1],pos[2],0,0,0,1,&tmp)
        jcb[1,2] = tmp

        self.itpz._ev_(t,pos[0],pos[1],pos[2],0,1,0,0,&tmp)
        jcb[2,0] = tmp
        self.itpz._ev_(t,pos[0],pos[1],pos[2],0,0,1,0,&tmp)
        jcb[2,1] = tmp
        self.itpz._ev_(t,pos[0],pos[1],pos[2],0,0,0,1,&tmp)
        jcb[2,2] = tmp

        return np.copy(jcb)
