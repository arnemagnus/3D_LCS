cimport cython

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

from libc.math cimport fmod as cy_fmod, floor as cy_floor, pow as cy_pow

from scipy.linalg.cython_blas cimport daxpy as cy_daxpy, ddot as cy_ddot, \
                                      dnrm2 as cy_dnrm2, dscal as cy_dscal, \
                                      dcopy as cy_dcopy

ctypedef vector[double] dbl_vec


cdef void _daxpy_(double alpha, dbl_vec &x, dbl_vec &y):
    cdef:
        int N = x.size()
        int INCX = 1, INCY = 1

    cy_daxpy(&N,
             &alpha,
             &x[0],
             &INCX,
             &y[0],
             &INCY)

cdef void _dscal_(double alpha, dbl_vec &x):
    cdef:
        int N = x.size()
        int INCX = 1

    cy_dscal(&N,
             &alpha,
             &x[0],
             &INCX)

cdef void _dcopy_(dbl_vec &x, dbl_vec &y):
    cdef:
        int N = x.size()
        int INCX = 1, INCY = 1
    "Copy X to Y"
    cy_dcopy(&N,
             &x[0],
             &INCX,
             &y[0],
             &INCY)

cdef void _dcopy_mv_vec_(double[::1] &x, dbl_vec &y):
    cdef:
        int N = y.size()
        int INCX = 1, INCY = 1
    " Copy X to Y"
    cy_dcopy(&N,
             &x[0],
             &INCX,
             &y[0],
             &INCY)

cdef double _ddot_(dbl_vec &x, dbl_vec &y):
    cdef:
        int N = x.size()
        int INCX = 1, INCY = 1

    return cy_ddot(&N,
                   &x[0],
                   &INCX,
                   &y[0],
                   &INCY)

cdef double _dnrm2_(dbl_vec &x):
    cdef:
        int N = x.size()
        int INCX = 1

    return cy_dnrm2(&N,
                    &x[0],
                    &INCX)


cdef class AimAssister:
    cdef:
        double[:,:,:,::1] xi
        dbl_vec target
        double x_min, x_max, y_min, y_max, z_min, z_max, dx, dy, dz
        int nx, ny, nz

    def __cinit__(self):
        self.target = dbl_vec(3)

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] target,
                       double[::1] x, double[::1] y, double[::1] z,
                       double[:,:,:,::1] xi):
        cdef:
            int ndim = target.shape[0]
            int nx = x.shape[0], ny = y.shape[0], nz = z.shape[0]
            int i

        if ndim != 3:
            raise RuntimeError('This AimAssister is tailor-made to be used for\
                                three-dimensional data!')
        if nx != xi.shape[0] or ny != xi.shape[1] or nz != xi.shape[2]:
            raise RuntimeError('Array dimensions not aligned!')

        for i in range(ndim):
            self.target[i] = target[i]

        self.x_min = x[0]
        self.x_max = x[nx-1]
        self.dx = x[1]-x[0]
        self.nx = nx
        self.y_min = y[0]
        self.y_max = y[ny-1]
        self.dy = y[1]-y[0]
        self.ny = ny
        self.z_min = z[0]
        self.z_max = z[nz-1]
        self.dz = z[1]-z[0]
        self.nz = nz
        self.xi = xi

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    def __call__(self, double t, double[::1] x):
        if x.shape[0] != 3:
            raise RuntimeError('This AimAssister is tailor-made to be used for\
                                three-dimensional data!')
        return self._evaluate_(x)

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef np.ndarray[ndim=1,dtype=np.float64_t] _evaluate_(self, double[::1] pos):
        cdef:
            np.ndarray[ndim=1,dtype=np.float64_t] ret = np.zeros(3)
            dbl_vec xi3_internal
            dbl_vec rvec
            int i
            double nrm, wt

        xi3_internal = dbl_vec(3)

        rvec = dbl_vec(3)

        for i in range(3):
            rvec[i] = self.target[i] - pos[i]

        self._compute_lin_approx_(pos, xi3_internal)

        wt = _ddot_(rvec,xi3_internal)

        _daxpy_(-wt,xi3_internal,rvec)

        _dscal_(cy_pow(_dnrm2_(rvec),-1),rvec)

        for i in range(3):
            ret[i] = rvec[i]

        return ret

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef void _compute_lin_approx_(self, double[::1] &pos, dbl_vec &ret):
        cdef:
            double x, y, z
            int ix, ixp1, iy, iyp1, iz, izp1
            dbl_vec c0, c1, c2, c3, c4, c5, c6, c7

        x, y, z = pos[0], pos[1], pos[2]

        self._compute_indices_and_weights_(&x,&ix,&ixp1,&y,&iy,&iyp1,&z,&iz,&izp1)

        c0 = dbl_vec(3)
        c1 = dbl_vec(3)
        c2 = dbl_vec(3)
        c3 = dbl_vec(3)
        c4 = dbl_vec(3)
        c5 = dbl_vec(3)
        c6 = dbl_vec(3)
        c7 = dbl_vec(3)

        self._set_corner_vectors_(ix, ixp1, iy, iyp1, iz, izp1,
                                c0, c1, c2, c3, c4, c5, c6, c7)

        self._compute_normalized_weighted_sum_(ret, x, y, z, c0, c1, c2, c3, c4, c5, c6, c7)

    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    cdef void _set_corner_vectors_(self,
                                   int &ix, int &ixp1,
                                   int &iy, int &iyp1,
                                   int &iz, int &izp1,
                                   dbl_vec &c0, dbl_vec &c1,
                                   dbl_vec &c2, dbl_vec &c3,
                                   dbl_vec &c4, dbl_vec &c5,
                                   dbl_vec &c6, dbl_vec &c7):
        cdef:
            int i

        _dcopy_mv_vec_(self.xi[ix,iy,iz],c0)
        _dcopy_mv_vec_(self.xi[ixp1,iy,iz],c1)
        _dcopy_mv_vec_(self.xi[ix,iyp1,iz],c2)
        _dcopy_mv_vec_(self.xi[ixp1,iyp1,iz],c3)
        _dcopy_mv_vec_(self.xi[ix,iy,izp1],c4)
        _dcopy_mv_vec_(self.xi[ixp1,iy,izp1],c5)
        _dcopy_mv_vec_(self.xi[ix,iyp1,izp1],c6)
        _dcopy_mv_vec_(self.xi[ixp1,iyp1,izp1],c7)

#        for i in range(3):
#            c0[i] = self.xi[ix,iy,iz,i]
#            c1[i] = self.xi[ixp1,iy,iz,i]
#            c2[i] = self.xi[ix,iyp1,iz,i]
#            c3[i] = self.xi[ixp1,iyp1,iz,i]
#            c4[i] = self.xi[ix,iy,izp1,i]
#            c5[i] = self.xi[ixp1,iy,izp1,i]
#            c6[i] = self.xi[ix,iyp1,izp1,i]
#            c7[i] = self.xi[ixp1,iyp1,izp1,i]

        # Fix orientation

        if _ddot_(c0, c1) < 0:
            _dscal_(-1, c1)
        if _ddot_(c0, c2) < 0:
            _dscal_(-1, c2)
        if _ddot_(c0, c3) < 0:
            _dscal_(-1, c3)
        if _ddot_(c0, c4) < 0:
            _dscal_(-1, c4)
        if _ddot_(c0, c5) < 0:
            _dscal_(-1, c5)
        if _ddot_(c0, c6) < 0:
            _dscal_(-1, c6)
        if _ddot_(c0, c7) < 0:
            _dscal_(-1, c7)

    cdef void _compute_normalized_weighted_sum_(self, dbl_vec &ret,
            double x, double y, double z, dbl_vec &c0, dbl_vec &c1, dbl_vec &c2,
            dbl_vec &c3, dbl_vec &c4, dbl_vec &c5, dbl_vec &c6, dbl_vec &c7):

        cdef:
            dbl_vec tmp

        # LC along z-axis
        tmp = dbl_vec(3)
        _daxpy_(z,c4,tmp)
        _dscal_((1-z),c0)
        _daxpy_(1.,tmp,c0)

        tmp = dbl_vec(3)
        _daxpy_(z,c5,tmp)
        _dscal_((1-z),c1)
        _daxpy_(1.,tmp,c1)

        tmp = dbl_vec(3)
        _daxpy_(z,c6,tmp)
        _dscal_((1-z),c2)
        _daxpy_(1.,tmp,c2)

        tmp = dbl_vec(3)
        _daxpy_(z,c7,tmp)
        _dscal_((1-z),c3)
        _daxpy_(1.,tmp,c3)

        # LC along y-axis
        tmp = dbl_vec(3)
        _daxpy_(y,c2,tmp)
        _dscal_((1-y),c0)
        _daxpy_(1.,tmp,c0)

        tmp = dbl_vec(3)
        _daxpy_(y,c3,tmp)
        _dscal_((1-y),c1)
        _daxpy_(1.,tmp,c1)

        # LC along x-axis
        tmp = dbl_vec(3)
        _daxpy_(x,c1,tmp)
        _dscal_((1-x),c0)
        _daxpy_(1.,tmp,c0)

        # Normalize
        _dscal_(cy_pow(_dnrm2_(c0),-1),c0)

        # Assign to output
        _dcopy_(c0,ret)

    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void _compute_indices_and_weights_(self, double *x, int *ix, int *ixp1,
                                                  double *y, int *iy, int *iyp1,
                                                  double *z, int *iz, int *izp1):

        x[0] = cy_fmod((x[0]-self.x_min)/self.dx, self.nx)
        y[0] = cy_fmod((y[0]-self.y_min)/self.dy, self.ny)
        z[0] = cy_fmod((z[0]-self.z_min)/self.dz, self.nz)

        while x[0] < 0:
            x[0] += self.nx
        while y[0] < 0:
            y[0] += self.ny
        while z[0] < 0:
            z[0] += self.nz

        ix[0] = int(cy_floor(x[0]))
        iy[0] = int(cy_floor(y[0]))
        iz[0] = int(cy_floor(z[0]))

        x[0] -= ix[0]
        y[0] -= iy[0]
        z[0] -= iz[0]

        ixp1[0] = int((ix[0]+1)%self.nx)
        iyp1[0] = int((iy[0]+1)%self.ny)
        izp1[0] = int((iz[0]+1)%self.nz)
