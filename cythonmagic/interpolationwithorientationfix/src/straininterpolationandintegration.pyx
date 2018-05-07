cimport cython
cimport numpy as np
import numpy as np


from libc.math cimport fmod as c_fmod, floor as c_floor, pow as c_pow, fabs as c_fabs, copysign as c_copysign

from scipy.linalg.cython_blas cimport daxpy as scp_daxpy, dcopy as scp_dcopy,\
                                      ddot as scp_ddot, dnrm2 as scp_dnrm2, \
                                      dscal as scp_dscal

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void daxpy(int N, double alpha, double[::1] &x, int INCX, double[::1] &y, int INCY) nogil:
    # y <- ax + y
    scp_daxpy(&N, &alpha, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dcopy(int N, double[::1] &x, int INCX, double[::1] &y, int INCY) nogil:
    # y <- x
    scp_dcopy(&N, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double ddot(int N, double[::1] &x, int INCX, double[::1] &y, int INCY) nogil:
    # return xT*y
    return scp_ddot(&N, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef double dnrm2(int N, double[::1] &x, int INCX) nogil:
    # return (xT*t)**0.5
    return scp_dnrm2(&N, &x[0], &INCX)

@cython.initializedcheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef void dscal(int N, double alpha, double[::1] &x, int INCX) nogil:
    # x <- alpha*x
    scp_dscal(&N, &alpha, &x[0], &INCX)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _cy_cross_product_(double[::1] u, double[::1] v, double[::1] &ret):
    # ret = u x v
    ret[0] = u[1]*v[2]-u[2]*v[1]
    ret[1] = -(u[0]*v[2]-u[2]*v[0])
    ret[2] = u[0]*v[1]-u[1]*v[0]

@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _cy_normalize_(double[::1] v):
    # v <- v / ||v||_2
    cdef:
        int N = v.shape[0]
        int INCX = 1
    while(dnrm2(N,v,INCX) < 0.0001):
        dscal(N,100,v,INCX)
    dscal(N,1/dnrm2(N,v,INCX),v,INCX)

# Exposing the interfaced Fortran functions (via a C++ class)
cdef extern from "linkto.h":
    cppclass ItpCont:
        ItpCont() except +
        void init_interp(double *x, double *y, double *z, double *f,
                int kx, int ky, int kz, int nx, int ny, int nz, int ext) except +
        double eval_interp(double x, double y, double z, int dx, int dy, int dz) except +
        void kill_interp() except +

# In-house wrapper for the bspline-3d Fortran class
@cython.internal
cdef class SplineInterpolator:
    cdef:
        ItpCont *interp
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
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
    cdef void _ev_(self, double x, double y, double z, int kx, int ky, int kz, int kq, double*ret):
        ret[0] = self.interp.eval_interp(x,y,z,kx,ky,kz)
    def __dealoc__(self):
        if self.interp is not NULL:
            del self.interp

# Hidden superclass, mirroring a virtual superclass in C++.
# Intended to be subclassed, and/or sent into other extension types defined
# below.
@cython.internal
cdef class InterpolatorWithOrientationFix:
    def __init__(self, double[::1] x not None, double[::1] y not None, double[::1] z not None,
                       double[:,:,:,::1] xi not None, int kx, int ky, int kz):
        pass
    def __call__(self, double[::1] pos not None):
        pass
    cdef void _interpolate_xi_(self, double[::1] pos, double[::1] &ret):
        pass
    def __dealoc__(self):
        pass

cdef class LinearSpecialInterpolator(InterpolatorWithOrientationFix):
    """
    A Cython extension type which facilitates linear interpolation of
    (stationary) vector fields in R^3, with a local orientation fix in order
    to combat orientational discontinuities, with normalized results.
    Periodic boundary conditions are assumed.

    Methods defined here:
    LinearSpecialInterpolator.__init__(x,y,z,xi)
    LinearSpecialInterpolator.__call__(pos)

    Version: 0.3

    """
    cdef:
        double[:,:,:,::1] xi
        double x_min, x_max, y_min, y_max, z_min, z_max
        double dx, dy, dz
        int nx, ny, nz
        double _xi_cube_[8][3]
        double[:,::1] xi_cube
        double _xia_[3]
        double[::1] xia
        double _xi_ref_[3]
        double[::1] xi_ref
        int ind_x, ind_xp1, ind_y, ind_yp1, ind_z, ind_zp1
        double _ret_[3]
        double[::1] ret
        double x_internal, y_internal, z_internal

    def __cinit__(self):
        self.xia = self._xia_
        self.xi_ref = self._xi_ref_
        self.xi_cube = self._xi_cube_
        self.ret = self._ret_

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] x not None, double[::1] y not None,
                       double[::1] z not None, double[:,:,:,::1] xi not None):
        """
        LinearSpecialInterpolator.__init__(x,y,z,xi)

        Constructor for a LinearSpecialInterpolator instance.

        Parameters
        ----------
        x : (nx,) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape (nx,),
           containing the sampling points along the x abscissa.
           Must be strictly increasing.
        y : (ny,) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape (ny,),
           containing the sampling points along the y abscissa.
           Must be strictly increasing.
        z : (nz,) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape (nz,),
           containing the sampling points along the z abscissa.
           Must be strictly increasing.
        xi : (nx,ny,nz,3) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape
           (nx,ny,nz,3), containing the sampled vector field.
           xi[i,j,k] should correspond to the sample at (x[i],y[j],z[k]).

        """
        if (xi.shape[0] != x.shape[0] or xi.shape[1] != y.shape[0] or xi.shape[2] != z.shape[0]):
            raise ValueError('Array dimensions not aligned!')

        if xi.shape[3] != 3:
            raise ValueError('The interpolator routine is custom-built for three-dimensional data!')

        # Enforcing periodic BC by not including the sampling points along the
        # last rows and columns:
        self.nx = x.shape[0] - 1
        self.ny = y.shape[0] - 1
        self.nz = z.shape[0] - 1

        self.x_min, self.x_max = x[0], x[self.nx-1]
        self.y_min, self.y_max = y[0], y[self.ny-1]
        self.z_min, self.z_max = z[0], z[self.nz-1]

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]

        self.xi = xi[:self.nx, :self.ny, :self.nz, :]

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __call__(self, double[::1] pos not None):
        """
        LinearSpecialInterpolator.__call__(pos)

        Evaluates the linearly interpolated vector field with orientation
        fix, at the coordinates specified by pos.

        Parameters
        ----------
        pos : (3,) array-like
           A (C-contiguous) NumPy array containing the (Cartesian) coordinates
           of the point in R^3 at which the vector field is to be interpolated.

        Returns
        -------
        xi : (3,) array-like
           A (C-contiguous) NumPy array containing the normalized, interpolated
           vector.

        """
        if pos.shape[0] != 3:
            raise ValueError('The interpolation routine is custom-built'\
                    +' for three-dimensional data!')
        self._interpolate_xi_(pos,self.ret)
        return np.copy(self.ret)


    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void _interpolate_xi_(self, double[::1] pos, double[::1] &xi):
        """
        LinearSpecialInterpolator._interpolate_xi_(pos,xi)

        The C-level function which performs the special interpolation.

        Parameters
        ----------
        pos : (3,) array-like
           A (C-contiguous) NumPy array containing the (Cartesian) coordinates
           of the point in R^3 at which the vector field is to be interpolated.
        xi : (3,) array-like, intent = out
           A (C-contiguous) NumPy array containing the normalized, interpolated
           vector.

        """
        cdef:
            double x = self.x_internal, y = self.y_internal, z = self.z_internal

        x = pos[0]
        y = pos[1]
        z = pos[2]

        self._compute_indcs_and_wts_(&x,&y,&z)

        self._set_crnr_vcs_()

        self._compute_nrmd_wtd_sum_(x,y,z,xi)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef void _compute_indcs_and_wts_(self, double *x, double *y, double *z):
        """
        LinearSpecialInterpolator._compute_indcs_and_wts_(x,y,z)

        The C-level function which determines the interpolation voxel
        and weights along each abscissa.

        Parameters
        ----------
        x : double, intent = inout
           On entry, the x-coordinate at which an interpolated vector is sought.
           On exit, the (normalized) interpolation weight along the x
           abscissa within the interpolation voxel.
        y : double, intent = inout
           On entry, the y-coordinate at which an interpolated vector is sought.
           On exit, the (normalized) interpolation weight along the y
           abscissa within the interpolation voxel.
        z : double, intent = inout
           On entry, the z-coordinate at which an interpolated vector is sought.
           On exit, the (normalized) interpolation weight along the z
           abscissa within the interpolation voxel.

        """

        x[0] = c_fmod((x[0]-self.x_min)/self.dx, self.nx)
        y[0] = c_fmod((y[0]-self.y_min)/self.dy, self.ny)
        z[0] = c_fmod((z[0]-self.z_min)/self.dz, self.nz)

        while x[0] < 0:
            x[0] += self.nx
        while y[0] < 0:
            y[0] += self.ny
        while z[0] < 0:
            z[0] += self.nz

        self.ind_x = int(c_floor(x[0]))
        self.ind_y = int(c_floor(y[0]))
        self.ind_z = int(c_floor(z[0]))

        x[0] -= self.ind_x
        y[0] -= self.ind_y
        z[0] -= self.ind_z

        self.ind_xp1 = int((self.ind_x+1)%self.nx)
        self.ind_yp1 = int((self.ind_y+1)%self.ny)
        self.ind_zp1 = int((self.ind_z+1)%self.nz)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _set_crnr_vcs_(self):
        """
        LinearSpecialInterpolator._set_crnr_vcs_()

        The C-level function which identifies the vector values at the
        corners of the interpolation voxel, generating local copies which
        are corrected for orientational discontinuities prior to
        linear interpolation.

        """
        cdef:
            double[::1] xi_tmp = self.xia, xi_ref = self.xi_ref
            double[:,:,:,::1] xi = self.xi
            double[:,::1] xi_cube = self.xi_cube

        dcopy(3,xi[self.ind_x,self.ind_y,self.ind_z],1,xi_ref,1)

        dcopy(3,xi[self.ind_x,self.ind_y,self.ind_z],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[0],1)

        dcopy(3,self.xi[self.ind_xp1,self.ind_y,self.ind_z],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[1],1)

        dcopy(3,self.xi[self.ind_x,self.ind_yp1,self.ind_z],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[2],1)

        dcopy(3,self.xi[self.ind_xp1,self.ind_yp1,self.ind_z],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[3],1)

        dcopy(3,self.xi[self.ind_x,self.ind_y,self.ind_zp1],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[4],1)

        dcopy(3,self.xi[self.ind_xp1,self.ind_y,self.ind_zp1],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[5],1)

        dcopy(3,self.xi[self.ind_x,self.ind_yp1,self.ind_zp1],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[6],1)

        dcopy(3,self.xi[self.ind_xp1,self.ind_yp1,self.ind_zp1],1,xi_tmp,1)
        if ddot(3,xi_tmp,1,xi_ref,1) < 0:
            dscal(3,-1,xi_tmp,1)
        dcopy(3,xi_tmp,1,xi_cube[7],1)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void _compute_nrmd_wtd_sum_(self, double x, double y, double z, double[::1] &xi):
        """
        LinearSpecialInterpolator._compute_nrmd_wtd_sum_(x,y,z,xi)

        The C-level function which computes the normalized, weighted sum
        which generates the interpolated vector.

        Parameters
        ----------
        x : double
           The (normalized) interpolation weight along the x abscissa within the
           interpolation voxel.
        y : double
           The (normalized) interpolation weight along the y abscissa within the
           interpolation voxel.
        z : double
           The (normalized) interpolation weight along the z abscissa within the
           interpolation voxel.
        xi : (3,) memoryview, intent = out
            On exit, the normalized, linearly interpolated vector, as a
            (C-contiguous) memoryview of doubles.

        """
        cdef:
            double[:,::1] xi_cube = self.xi_cube
        # LC along z-axis
        dscal(3,1-z,xi_cube[0],1)
        daxpy(3,z,xi_cube[4],1,xi_cube[0],1)

        dscal(3,1-z,xi_cube[1],1)
        daxpy(3,z,xi_cube[5],1,xi_cube[1],1)

        dscal(3,1-z,xi_cube[2],1)
        daxpy(3,z,xi_cube[6],1,xi_cube[2],1)

        dscal(3,1-z,xi_cube[3],1)
        daxpy(3,z,xi_cube[7],1,xi_cube[3],1)

        # LC along y-axis
        dscal(3,1-y,xi_cube[0],1)
        daxpy(3,y,xi_cube[2],1,xi_cube[0],1)

        dscal(3,1-y,xi_cube[1],1)
        daxpy(3,y,xi_cube[3],1,xi_cube[1],1)

        # LC along x-axis
        dscal(3,1-x,xi_cube[0],1)
        daxpy(3,x,xi_cube[1],1,xi_cube[0],1)

        # Assign to output
        dcopy(3,xi_cube[0],1,xi,1)

        # Normalize output
        _cy_normalize_(xi)

    def __dealoc__(self):
        pass

cdef class CubicSpecialInterpolator(InterpolatorWithOrientationFix):
    """
    A Cython extension type which facilitates cubic B-spline interpolation of
    (stationary) vector fields in R^3, with a local orientation fix in order
    to combat orientational discontinuities, with normalized results.
    Periodic boundary conditions are assumed.

    NOTE: In order for this to function as intended (allowing for cubic
          polylomial pieces, corresponding to a quartic spline), the
          bspline-fortran source code must be slightly modified.
          In particular, in the subroutine 'check_k', which is located withihn
          bspline_sub_module.f90, the second condition of the if statement
          must be changed from 'k >= n' to 'k > n'.

    Methods defined here:
    CubicSpecialInterpolator.__init__(x,y,z,xi)
    CubicSpecialInterpolator.__call__(pos)

    Version: 0.3

    """
    cdef:
        ItpCont *itpx
        ItpCont *itpy
        ItpCont *itpz
        double[:,:,:,::1] xi_grid
        double[::1] x_grid, y_grid, z_grid
        double dx, dy, dz
        int nx, ny, nz
        double _x_cube_[4]
        double _y_cube_[4]
        double _z_cube_[4]
        double[::1] x_cube, y_cube, z_cube
        int _inds_x_[4]
        int _inds_y_[4]
        int _inds_z_[4]
        int[::1] inds_x, inds_y, inds_z
        double _xi_cube_[3][64]
        double[:,::1] xi_cube
        double _xival_[3]
        double[::1] xival
        double _tmp_[3]
        double[::1] tmp
        double _xi_ref_[3]
        double[::1] xi_ref
        double _ret_[3]
        double[::1] ret
        bint calibrated, sameaslast
        int kx, ky, kz
        double x_internal, y_internal, z_internal

    def __cinit__(self):
        self.itpx = new ItpCont()
        self.itpy = new ItpCont()
        self.itpz = new ItpCont()
        self.x_cube = self._x_cube_
        self.y_cube = self._y_cube_
        self.z_cube = self._z_cube_
        self.inds_x = self._inds_x_
        self.inds_y = self._inds_y_
        self.inds_z = self._inds_z_
        self.xi_cube = self._xi_cube_
        self.tmp = self._tmp_
        self.xi_ref = self._xi_ref_
        self.calibrated = False
        self.ret = self._ret_
        self.sameaslast = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] x not None, double[::1] y not None, double[::1] z not None, \
            double[:,:,:,::1] xi not None, int kx = 4, int ky = 4, int kz = 4):
        """
        CubicSpecialInterpolator.__init__(x,y,z,xi)

        Constructor for a CubicSpecialInterpolator instance.

        Parameters
        ----------
        x : (nx,) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape (nx,),
           containing the sampling points along the x abscissa.
           Must be strictly increasing.
        y : (ny,) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape (ny,),
           containing the sampling points along the y abscissa.
           Must be strictly increasing.
        z : (nz,) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape (nz,),
           containing the sampling points along the z abscissa.
           Must be strictly increasing.
        xi : (nx,ny,nz,3) array-like
           A (C-contiguous) NumPy array of type np.float64 and shape
           (nx,ny,nz,3), containing the sampled vector field.
           xi[i,j,k] should correspond to the sample at (x[i],y[j],z[k]).
        kx : integer, optional
           Spline order along the x abscissa. Default: kx = 4
        ky : integer, optional
           Spline order along the y abscissa. Default: ky = 4
        kz : integer, optional
           Spline order along the z abscissa. Default: kz = 4

        """
        cdef:
            int i
        if(xi.shape[0] != x.shape[0] or xi.shape[1] != y.shape[0] or xi.shape[2] != z.shape[0]):
            raise ValueError('Array dimensions not aligned!')
        if (x.shape[0] < 4 or y.shape[0] < 4 or z.shape[0] < 4):
            raise ValueError('Insufficient amount of data points to perform cubic interpolation!')
        if(xi.shape[3] != 3):
            raise ValueError('The interpolator routine is custom-built for three dimensional data!')

        if (kx < 2 or kx > 4) or (ky < 2 or ky > 4) or (kz < 2 or kz > 4):
            raise ValueError('Invalid choice of interpolator order!')
        # Enforcing periodic BC by not including the sampling points along
        # the last rows and columns
        self.nx = x.shape[0]-1
        self.ny = y.shape[0]-1
        self.nz = z.shape[0]-1

        self.x_grid = x[:self.nx]
        self.y_grid = y[:self.ny]
        self.z_grid = z[:self.nz]

        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0]

        # Set local coordinates for interpolation within any given 4x4x4 voxel
        for i in range(4):
            self.x_cube[i] = (i-1)*self.dx
            self.y_cube[i] = (i-1)*self.dy
            self.z_cube[i] = (i-1)*self.dz

        self.xi_grid = xi[:self.nx,:self.ny,:self.nz,:]

        self.kx = kx
        self.ky = ky
        self.kz = kz

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double[::1] pos not None):
        """
        CubicSpecialInterpolator.__call__(pos)

        Evaluates the B-spline interpolated vector field with orientation
        fix, at the coordinates specified by pos.

        Parameters
        ----------
        pos : (3,) array-like
           A (C-contiguous) NumPy array containing the (Cartesian) coordinates
           of the point in R^3 at which the vector field is to be interpolated.

        Returns
        -------
        xi : (3,) array-like
           A (C-contiguous) NumPy array containing the normalized, interpolated
           vector.

        """
        if pos.shape[0] != 3:
            raise ValueError('The interpolation routine is custom-built'\
                    +' for three dimensional data')
        self._interpolate_xi_(pos,self.ret)
        return np.copy(self.ret)


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _interpolate_xi_(self, double[::1] pos, double[::1] &xi):
        """
        CubicSpecialInterpolator._interpolate_xi_(pos,xi)

        The C-level function which performs the special interpolation.

        Parameters
        ----------
        pos : (3,) array-like
           A (C-contiguous) NumPy array containing the (Cartesian) coordinates
           of the point in R^3 at which the vector field is to be interpolated.
        xi : (3,) array-like, intent = out
           A (C-contiguous) NumPy array containing the normalized, interpolated
           vector.

        """
        cdef:
            double x = self.x_internal, y = self.y_internal, z = self.z_internal
            int i, j, k,l
            int[::1] inds_x = self.inds_x, inds_y = self.inds_y, inds_z = self.inds_z
            double[::1] x_cube = self.x_cube, y_cube = self.y_cube, z_cube = self.z_cube
            double[:,::1] xi_cube = self.xi_cube
            double[:,:,:,::1] xi_grid = self.xi_grid
            double[::1] tmp_xi = self.tmp, xi_ref = self.xi_ref
        x, y, z = pos[0], pos[1], pos[2]
        while x < self.x_grid[0]:
            x += self.x_grid[self.nx-1] - self.x_grid[0]
        while y < self.y_grid[0]:
            y += self.y_grid[self.ny-1] - self.y_grid[0]
        while z < self.z_grid[0]:
            z += self.z_grid[self.nz-1] - self.z_grid[0]

        while x > self.x_grid[self.nx-1]:
            x -= (self.x_grid[self.nx-1]-self.x_grid[0])
        while y > self.y_grid[self.ny-1]:
            y -= (self.y_grid[self.ny-1]-self.y_grid[0])
        while z > self.z_grid[self.nz-1]:
            z -= (self.z_grid[self.nz-1]-self.z_grid[0])

        self._set_voxel_indices_(x,y,z)
        x -= self.x_grid[inds_x[1]]
        y -= self.y_grid[inds_y[1]]
        z -= self.z_grid[inds_z[1]]
        if not (self.sameaslast and self.calibrated):
            dcopy(3,xi_grid[inds_x[0],inds_y[0],inds_z[0]],1,xi_ref,1)
            for k in range(4):
                for j in range(4):
                    for i in range(4):
                        dcopy(3,xi_grid[inds_x[i],inds_y[j],inds_z[k]],1,tmp_xi,1)
                        if ddot(3,tmp_xi,1,xi_ref,1) < 0:
                            for l in range(3):
                                xi_cube[l,i+4*(j+4*k)] = -tmp_xi[l]
                        else:
                            for l in range(3):
                                xi_cube[l,i+4*(j+4*k)] = tmp_xi[l]

            self.itpx.init_interp(&x_cube[0],&y_cube[0],&z_cube[0],&xi_cube[0,0],self.kx,self.ky,self.kz,4,4,4,0)
            self.itpy.init_interp(&x_cube[0],&y_cube[0],&z_cube[0],&xi_cube[1,0],self.kx,self.ky,self.kz,4,4,4,0)
            self.itpz.init_interp(&x_cube[0],&y_cube[0],&z_cube[0],&xi_cube[2,0],self.kx,self.ky,self.kz,4,4,4,0)
            self.calibrated = True
        # Evaluate interpolation objects
        xi[0] = self.itpx.eval_interp(x,y,z,0,0,0)
        xi[1] = self.itpy.eval_interp(x,y,z,0,0,0)
        xi[2] = self.itpz.eval_interp(x,y,z,0,0,0)
        # Normalize results
        _cy_normalize_(xi)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void _set_voxel_indices_(self,double x, double y, double z):
        """
        CubicSpecialInterpolator._set_voxel_indices_(x,y,z)

        The C-level function which determines the interpolation voxel.

        Parameters
        ----------
        x : double
           The x-coordinate at which an interpolated vector is sought.
        y : double
           The y-coordinate at which an interpolated vector is sought.
        z : d ouble
           The z-coordinate at which an interpolated vector is sought.

        """
        cdef:
            int[::1] inds_x = self.inds_x, inds_y = self.inds_y, inds_z = self.inds_z
            int ind_x, ind_y, ind_z

        x = c_fmod((x-self.x_grid[0])/(self.dx),self.nx)
        y = c_fmod((y-self.y_grid[0])/(self.dy),self.ny)
        z = c_fmod((z-self.z_grid[0])/(self.dz),self.nz)

        while x < 0:
            x += self.nx
        while y < 0:
            y += self.ny
        while z < 0:
            z += self.nz

        ind_x = int(c_floor(x))
        ind_y = int(c_floor(y))
        ind_z = int(c_floor(z))

        if not self.calibrated or (self.inds_x[1] != ind_x or self.inds_y[1] != ind_y
                                    or self.inds_z[1] != ind_z):

            inds_x[1] = ind_x
            inds_y[1] = ind_y
            inds_z[1] = ind_z

            inds_x[0] = (ind_x-1)%(self.nx)
            inds_x[2] = (ind_x+1)%(self.nx)
            inds_x[3] = (ind_x+2)%(self.nx)

            while inds_x[0] < 0:
                inds_x[0] += self.nx
            while inds_x[2] < 0:
                inds_x[2] += self.nx
            while inds_x[3] < 0:
                inds_x[3] += self.nx

            inds_y[0] = (ind_y-1)%(self.ny)
            inds_y[2] = (ind_y+1)%(self.ny)
            inds_y[3] = (ind_y+2)%(self.ny)

            while inds_y[0] < 0:
                inds_y[0] += self.ny
            while inds_y[2] < 0:
                inds_y[2] += self.ny
            while inds_y[3] < 0:
                inds_y[3] += self.ny

            inds_z[0] = (ind_z-1)%(self.nz)
            inds_z[2] = (ind_z+1)%(self.nz)
            inds_z[3] = (ind_z+2)%(self.nz)

            while inds_z[0] < 0:
                inds_z[0] += self.nz
            while inds_z[2] < 0:
                inds_z[2] += self.nz
            while inds_z[3] < 0:
                inds_z[3] += self.nz

            self.sameaslast = False
        else:
            self.sameaslast = True

    def __dealoc__(self):
        if self.itpx is not NULL:
            del self.itpx
        if self.itpy is not NULL:
            del self.itpy
        if self.itpz is not NULL:
            del self.itpz


cdef class StrainAimAssister:
    cdef:
        double _tan_vec_[3]
        double _prev_vec_[3]
        double _xi_[3]
        double _ret_[3]
        double[::1] tan_vec, prev_vec, xi, ret
        InterpolatorWithOrientationFix xi_itp
        bint initialized_tan, initialized_prev

    def __cinit__(self):
        self.tan_vec = self._tan_vec_
        self.prev_vec = self._prev_vec_
        self.xi = self._xi_
        self.ret = self._ret_
        self.initialized_tan = False
        self.initialized_prev = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, InterpolatorWithOrientationFix xi_itp):
        self.xi_itp = xi_itp

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_tan_vec(self, double[::1] tan_vec not None):
        if tan_vec.shape[0] != 3:
            raise ValueError('The interpolation-aiming routine is custom-built' \
                    + ' for three-dimensional data!')

        dcopy(3,tan_vec,1,self.tan_vec,1)
        _cy_normalize_(self.tan_vec)
        self.initialized_tan = True

    def unset_tan_vec(self):
        self.initialized_tan = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_prev_vec(self, double[::1] prev_vec not None):
        if prev_vec.shape[0] != 3:
            raise ValueError('The interpolation-aiming routine is custom-built' \
                    + ' for three-dimensional data!')

        dcopy(3,prev_vec,1,self.prev_vec,1)
        _cy_normalize_(self.prev_vec)
        self.initialized_prev = True

    def unset_prev_vec(self):
        self.initialized_prev = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos not None):
        if pos.shape[0] != 3:
            raise ValueError('The interpolation-aiming routine is custom-built' \
                    + ' for three dimensional data!')

        if not (self.initialized_tan and self.initialized_prev):
            raise RuntimeError('Aim assister not initialized with target!')
        self._ev_(pos,self.ret)
        return np.copy(self.ret)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void _ev_(self, double[::1] pos, double[::1] &ret):
        cdef:
            double[::1] xi = self.xi

        self.xi_itp._interpolate_xi_(pos, xi)
        if 1 - c_pow(ddot(3,xi,1,self.tan_vec,1),2) < 1e-8:
            dcopy(3,self.prev_vec,1,ret,1)
        else:
            _cy_cross_product_(xi,self.tan_vec,ret)
        _cy_normalize_(ret)
        if ddot(3,ret,1,self.prev_vec,1) < 0:
            dscal(3,-1,ret,1)

    def __dealoc__(self):
        pass


cdef class Dp87Strain:
    cdef:
        double fac, maxfac
        double _c_[12]
        double[::1] c
        double _a1_[1]
        double _a2_[2]
        double _a3_[3]
        double _a4_[4]
        double _a5_[5]
        double _a6_[6]
        double _a7_[7]
        double _a8_[8]
        double _a9_[9]
        double _a10_[10]
        double _a11_[11]
        double _a12_[12]
        double _a13_[13]
        double[::1] a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13
        double _b7_[13]
        double _b8_[13]
        double[::1] b7, b8
        double _x7_[3]
        double _x8_[3]
        double[::1] x7, x8
        double _k1_[3]
        double _k2_[3]
        double _k3_[3]
        double _k4_[3]
        double _k5_[3]
        double _k6_[3]
        double _k7_[3]
        double _k8_[3]
        double _k9_[3]
        double _k10_[3]
        double _k11_[3]
        double _k12_[3]
        double _k13_[3]

        double[::1] k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, k13
        double _k_tmp_[3]
        double[::1] k_tmp
        double _pos_i_[3]
        double[::1] pos_i
        double tmp, sc, err, h_opt
        double atol, rtol
        double q
        StrainAimAssister f
        bint initialized

    def __cinit__(self):
        self._c_  = [0.05555555555555555556, 0.08333333333333333333, 0.125, 0.3125, 0.375, 0.1475, 0.465, 0.56486545138225957359, 0.65, 0.92465627764050444674, 1., 1.]
        self._a1_ = [0.05555555555555555556]
        self._a2_ = [0.0208333333333333333, 0.0625]
        self._a3_ = [0.03125, 0., 0.09375]
        self._a4_ = [0.3125, 0., -1.17185, 1.17185]
        self._a5_ = [0.0375, 0., 0., 0.1875, 0.15]
        self._a6_ = [0.04791013711111111111, 0., 0., 0.11224871277777777777, -0.02550567377777777777, 0.012846823888888888]
        self._a7_ = [0.016917989787292281, 0., 0., 0.387848278486043169, 0.035977369851500327, 0.196970214215666060, -0.172713852340501838]
        self._a8_ = [0.069095753359192300, 0., 0., -0.63424797672885411, -0.16119757522460408, 0.138650309458825255, 0.940928614035756269, 0.211636326481943981]
        self._a9_ = [0.183556996839045385, 0., 0., -2.46876808431559245, -0.29128688781630045, -0.02647302023311737, 2.847838764192800449, 0.281387331469849792, 0.123744899863314657]
        self._a10_ = [-1.21542481739588805, 0., 0., 16.672608665945772432, 0.915741828416817960, -6.05660580435747094, -16.00357359415617811, 14.849303086297662557, -13.371575735289849318, 5.134182648179637933]
        self._a11_ = [0.258860916438264283, 0., 0., -4.774485785489205112, -0.435093013777032509, -3.049483332072241509, 5.577920039936099117, 6.155831589861040689, -5.062104586736938370, 2.193926173180679061, 0.134627998659334941]
        self._a12_ = [0.822427599626507477, 0., 0.,  -11.658673257277664283, -0.757622116690936195, 0.713973588159581527, 12.075774986890056739, -2.127659113920402656, 1.990166207048955418, -0.234286471544040292, 0.175898577707942265, 0.  ]

        self._b7_ = [ 0.0295532136763534969, 0., 0., 0., 0., -0.8286062764877970397, 0.3112409000511183279, 2.4673451905998869819, -2.5469416518419087391, 1.4435485836767752403, 0.0794155958811272872, 0.0444444444444444444, 0. ]
        self._b8_ = [ 0.0417474911415302462, 0., 0., 0., 0., -0.0554523286112393089, 0.2393128072011800970, 0.7035106694034430230, -0.7597596138144609298, 0.6605630309222863414, 0.1581874825101233355, -0.2381095387528628044, 0.25]


        self.c = self._c_
        self.a1 = self._a1_
        self.a2 = self._a2_
        self.a3 = self._a3_
        self.a4 = self._a4_
        self.a5 = self._a5_
        self.a6 = self._a6_
        self.a7 = self._a7_
        self.a8 = self._a8_
        self.a9 = self._a9_
        self.a10 = self._a10_
        self.a11 = self._a11_
        self.a12 = self._a12_
        self.b7 = self._b7_
        self.b8 = self._b8_
        self.x7 = self._x7_
        self.x8 = self._x8_

        self.k1 = self._k1_
        self.k2 = self._k2_
        self.k3 = self._k3_
        self.k4 = self._k4_
        self.k5 = self._k5_
        self.k6 = self._k6_
        self.k7 = self._k7_
        self.k8 = self._k8_
        self.k9 = self._k9_
        self.k10 = self._k10_
        self.k11 = self._k11_
        self.k12 = self._k12_
        self.k13 = self._k13_
        self.k_tmp = self._k_tmp_

        self.pos_i = self._pos_i_

        self.fac = 0.8
        self.maxfac = 2.0
        self.q = 7.

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double atol, double rtol):
        self.atol = atol
        self.rtol = rtol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_aim_assister(self, StrainAimAssister direction_generator):
        self.f = direction_generator
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def unset_aim_assister(self):
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos not None, double h):
        cdef:
            double[::1] pos_i = self.pos_i
        if not self.initialized:
            raise RuntimeError('Dormand-Prince 8(7) strain solver not'\
                    ' initialized with a StrainAimAssister instance!')
        dcopy(3,pos,1,pos_i,1)
        self._ev_(&t,pos_i,&h)
        return t, np.copy(pos_i), h

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _ev_(self, double *t, double[::1] pos, double *h):
        cdef:
            double[::1] k1 = self.k1, k2 = self.k2, k3 = self.k3, k4 = self.k4, k5 = self.k5, k6 = self.k6, k7 = self.k7, k8 = self.k8, k9 = self.k9, k10 = self.k10, k11 = self.k11, k12 = self.k12, k13 = self.k13, k_tmp = self.k_tmp
            double[::1] x7 = self.x7, x8 = self.x8
            double[::1] b7 = self.b7, b8 = self.b8
            double[::1] a1 = self.a1, a2 = self.a2, a3 = self.a3, a4 = self.a4, a5 = self.a5, a6 = self.a6, a7 = self.a7, a8 = self.a8, a9 = self.a9, a10 = self.a10, a11 = self.a11, a12 = self.a12, a13 = self.a13
            int i

        self.f._ev_(pos,k1)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a1[0]*h[0],k1,1,k_tmp,1)

        self.f._ev_(k_tmp,k2)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a2[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a2[1]*h[0],k2,1,k_tmp,1)

        self.f._ev_(k_tmp,k3)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a3[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a3[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a3[2]*h[0],k3,1,k_tmp,1)

        self.f._ev_(k_tmp,k4)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a4[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a4[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a4[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a4[3]*h[0],k4,1,k_tmp,1)

        self.f._ev_(k_tmp,k5)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a5[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a5[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a5[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a5[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a5[4]*h[0],k5,1,k_tmp,1)

        self.f._ev_(k_tmp,k6)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a6[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a6[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a6[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a6[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a6[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a6[5]*h[0],k6,1,k_tmp,1)

        self.f._ev_(k_tmp,k7)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a7[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a7[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a7[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a7[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a7[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a7[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a7[6]*h[0],k7,1,k_tmp,1)

        self.f._ev_(k_tmp,k8)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a8[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a8[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a8[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a8[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a8[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a8[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a8[6]*h[0],k7,1,k_tmp,1)
        daxpy(3,a8[7]*h[0],k8,1,k_tmp,1)

        self.f._ev_(k_tmp,k9)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a9[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a9[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a9[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a9[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a9[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a9[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a9[6]*h[0],k7,1,k_tmp,1)
        daxpy(3,a9[7]*h[0],k8,1,k_tmp,1)
        daxpy(3,a9[8]*h[0],k9,1,k_tmp,1)

        self.f._ev_(k_tmp,k10)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a10[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a10[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a10[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a10[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a10[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a10[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a10[6]*h[0],k7,1,k_tmp,1)
        daxpy(3,a10[7]*h[0],k8,1,k_tmp,1)
        daxpy(3,a10[8]*h[0],k9,1,k_tmp,1)
        daxpy(3,a10[9]*h[0],k10,1,k_tmp,1)

        self.f._ev_(k_tmp,k11)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a11[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a11[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a11[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a11[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a11[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a11[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a11[6]*h[0],k7,1,k_tmp,1)
        daxpy(3,a11[7]*h[0],k8,1,k_tmp,1)
        daxpy(3,a11[8]*h[0],k9,1,k_tmp,1)
        daxpy(3,a11[9]*h[0],k10,1,k_tmp,1)
        daxpy(3,a11[10]*h[0],k11,1,k_tmp,1)

        self.f._ev_(k_tmp,k12)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a12[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a12[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a12[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a12[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a12[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a12[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a12[6]*h[0],k7,1,k_tmp,1)
        daxpy(3,a12[7]*h[0],k8,1,k_tmp,1)
        daxpy(3,a12[8]*h[0],k9,1,k_tmp,1)
        daxpy(3,a12[9]*h[0],k10,1,k_tmp,1)
        daxpy(3,a12[10]*h[0],k11,1,k_tmp,1)
        daxpy(3,a12[11]*h[0],k12,1,k_tmp,1)

        self.f._ev_(k_tmp,k13)

        dscal(3,0,x7,1)
        dscal(3,0,x8,1)

        daxpy(3,b7[0],k1,1,x7,1)
        daxpy(3,b7[1],k2,1,x7,1)
        daxpy(3,b7[2],k3,1,x7,1)
        daxpy(3,b7[3],k4,1,x7,1)
        daxpy(3,b7[4],k5,1,x7,1)
        daxpy(3,b7[5],k6,1,x7,1)
        daxpy(3,b7[6],k7,1,x7,1)
        daxpy(3,b7[7],k8,1,x7,1)
        daxpy(3,b7[8],k9,1,x7,1)
        daxpy(3,b7[9],k10,1,x7,1)
        daxpy(3,b7[10],k11,1,x7,1)
        daxpy(3,b7[11],k12,1,x7,1)
        daxpy(3,b7[12],k13,1,x7,1)

        dscal(3,h[0],x7,1)
        daxpy(3,1,pos,1,x7,1)

        daxpy(3,b8[0],k1,1,x8,1)
        daxpy(3,b8[1],k2,1,x8,1)
        daxpy(3,b8[2],k3,1,x8,1)
        daxpy(3,b8[3],k4,1,x8,1)
        daxpy(3,b8[4],k5,1,x8,1)
        daxpy(3,b8[5],k6,1,x8,1)
        daxpy(3,b8[6],k7,1,x8,1)
        daxpy(3,b8[7],k8,1,x8,1)
        daxpy(3,b8[8],k9,1,x8,1)
        daxpy(3,b8[9],k10,1,x8,1)
        daxpy(3,b8[10],k11,1,x8,1)
        daxpy(3,b8[11],k12,1,x8,1)
        daxpy(3,b8[12],k13,1,x8,1)

        dscal(3,h[0],x8,1)
        daxpy(3,1,pos,1,x8,1)

        if c_fabs(x8[0]) < c_fabs(x8[0]):
            self.sc = self.atol + c_fabs(x7[0])*self.rtol
        else:
            self.sc = self.atol + c_fabs(x8[0])*self.rtol

        self.err = c_fabs(x7[0]-x8[0])/self.sc
        for i in range(1,3):
            if c_fabs(x7[i]) < c_fabs(x8[i]):
                self.sc = self.atol + c_fabs(x7[i])*self.rtol
            else:
                self.sc = self.atol + c_fabs(x8[i])*self.rtol

            self.tmp = c_fabs(x7[i]-x8[i])/self.sc

            if self.tmp > self.err:
                self.err = self.tmp

        if self.err == 0:
            self.h_opt = c_copysign(10,h[0])
        else:
            self.h_opt = h[0]*c_pow((1/self.err),1/(self.q+1))

        if self.err < 1:
            t[0] = t[0] + h[0]
            dcopy(3,x8,1,pos,1)
            if self.maxfac*h[0] < self.fac*self.h_opt:
                h[0] = self.maxfac*h[0]
            else:
                h[0] = self.fac*self.h_opt
        else:
            h[0] = self.fac*self.h_opt
