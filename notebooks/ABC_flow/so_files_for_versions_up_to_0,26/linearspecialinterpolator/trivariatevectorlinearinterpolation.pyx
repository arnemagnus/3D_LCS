"""This module contains a Cython extension type which facilitates trilinear
interpolation of trivariate vector fields in R^3.

Furthermore, a Cython extension type which computes a unit normalized orthogonal
projection of a vector between two points into the surface which locally can be
approximated as a plane orthogonal to a trivariate vector field in R^3.

Lastly, a Cython extension type which arranges for Dormand-Prince 5(4)
approximation of trajectories which are defined as being orthogonal
to a trivariate vector field in R^3

Extension types defined here:
    LinearEigenvectorInterpolator
    LinearAimAssister
    Dp54Linear

Written by Arne Magnus T. Løken as part of my master's thesis work in physics
at NTNU, spring 2018.

"""


cimport cython

import time

import numpy as np
cimport numpy as np


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


cdef class LinearEigenvectorInterpolator:
    """A Cython extension type tailor-made for computing special-purpose
    linear interpolation of a three-dimensional, trivariate vector field,
    subject to periodic boundary conditions, with a local orientation fix prior
    to interpolation in order to combat local orientational discontinuities
    which may arise due to numerical noise.

    Methods defined here:
    LinearEigenvectorInterpolator.__init__(x,y,z,xi)
    LinearEigenvectorInterpolator.__call__(pos)

    Version: 0.2

    """
    cdef:
        double[:,:,:,::1] xi
        double x_min, x_max, y_min, y_max, z_min, z_max
        double dx, dy, dz
        int nx, ny, nz
        double _xi_ret_[3]
        double[::1] xi_ret
        double _xi_cube_[8][3]
        double[:,::1] xi_cube
        double _xia_[3]
        double[::1] xia
        double _xi_ref_[3]
        double[::1] xi_ref
        int ind_x, ind_xp1, ind_y, ind_yp1, ind_z, ind_zp1
        double _pos_[3]
        double[::1] pos
        double[::1] ret_mv
    def __cinit__(self):
        self.xi_ret = self._xi_ret_
        self.xia = self._xia_
        self.xi_ref = self._xi_ref_
        self.xi_ret = self._xi_ret_
        self.xi_cube = self._xi_cube_
        self.pos = self._pos_
        self.ret_mv = np.empty(3)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] x not None, double[::1] y not None, double[::1] z not None,\
            double[:,:,:,::1] xi not None):
        """LinearEigenvectorInterpolator(x,y,z,xi)

        Constructor for the three-dimensional, trivariate linear interpolation
        extension type. Periodic boundary conditions are enforced.

        param: x  -- Sampling points along the x abscissa, as a (C-contiguous)
                     NumPy array of shape (nx >= 2) and type np.float64.
        param: y  -- Sampling points along the y abscissa, as a (C-contiguous)
                     NumPy array of shape (ny >= 2) and type np.float64.
        param: z  -- Sampling points along the z abscissa, as a (C-contiguous)
                     NumPy array of shape (nz >= 2) and type np.float64.
        param: xi -- Sampled vector field at the grid spanned by the input
                     arrays x, y and z, as a (C-contiguous) NumPy array of shape
                     (nx,ny,nz,3) and type np.float64.

        """

        if (xi.shape[0] != x.shape[0] or xi.shape[1] != y.shape[0] or xi.shape[2] != z.shape[0]):
            raise RuntimeError('Array dimensions not aligned!')

        if xi.shape[3] != 3:
            raise RuntimeError('The interpolator routine is custom-built for three-dimensional data!')

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
        """LinearEigenvectorInterpolator.__call__(pos)

        Computes a linear interpolation of the vector field at 'pos', based
        upon the 2x2x2 set of nearest neighbor voxels, including a local
        direction fix --- i.e., ensuring that no pair of vectors is rotated
        more than 90 degrees with respect to eachother.

        Periodic boundary conditions are built in.

        param: pos -- Three-component (C-contiguous) NumPy array, containing
                      the Cartesian coordinates at which a linearly interpolated
                      vector is sought.

        return: vec -- Three-component (C-contiguous) NumPy array, containing
                       the aforementioned, normalized vector.

        """
        return np.asarray(self._ev_(pos))

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef double[::1] _ev_(self, double[::1] pos_):
        """LinearEigenvectorInterpolator._ev_(pos)

        The C-level function which computes the linearly interpolated vector
        field, which is returned by the __call__ routine.

        This function facilitates C-level computations with as little Python
        overhead as possible for related extension types (cf. LinearAimAssister,
        Dp54Linear).

        param: pos_ -- Three-component C-contiguous memoryview of doubles,
                       containing the Cartesian coordinates at which a spline
                       interpolated vector is sought

        return: vec -- Three-component C-contiguous memoryview of doubles,
                       containing the normalized Cartesian coordinates of the
                       spline interpolated eigenvector.

        """
        cdef:
            double[::1] ret_mv = self.ret_mv, xi = self.xi_ret
            double[::1] pos = self.pos

        if pos_.shape[0] != 3:
            raise RuntimeError('The interpolation routine is custom-build'\
                    +' for three-dimensional data!')

        dcopy(3,pos_,1,pos,1)

        self._interpolate_xi_(pos[0],pos[1],pos[2],xi)
        dcopy(3,xi,1,ret_mv,1)

        return ret_mv

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void _interpolate_xi_(self,double x, double y, double z, double[::1] &xi):
        """LinearEigenvectorInterpolator._interpolate_xi_(x,y,z,xi)

        The C-level function which computes the linearly interpolated vector
        field, which is returned by the __call__ routine.

        This function facilitates C-level computations with as little Python
        overhead as possible for related extension types (cf. LinearAimAssister
        Dp54Linear).

        param: x -- x-coordinate at which an interpolated vector is sought
        param: y -- y-coordinate at which an interpolated vector is sought
        param: z -- z-coordinate at which an interpolated vector is sought

        param/return: xi -- Three-component C-contiguous double memoryview,
                            which upon exit is overwritten with the interpolated
                            eigenvector.

        """

        self._compute_indices_and_weights_(&x,&y,&z)

        self._set_corner_vectors_()

        self._compute_normalized_weighted_sum_(x,y,z,xi)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.boundscheck(False)
    cdef void _compute_indices_and_weights_(self, double *x, double *y, double *z):
        """LinearEigenvectorInterpolator._set_voxel_indices_(x,y,z,xi)

        A C-level function which computes the indices of the points within
        the interpolation voxel which surrounds the point specified by the
        Cartesian coordinates [x,y,z].

        param: x -- x-coordinate at which an interpolated vector is sought
        param: y -- y-coordinate at which an interpolated vector is sought
        param: z -- z-coordinate at which an interpolated vector is sought

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
    cdef void _set_corner_vectors_(self):
        """LinearEigenvectorInterpolator._set_corner_vectors_()

        A C-level function which identifies the vectors at the corners of the
        interpolation voxel in question, performs a local orientation fix
        and places the (reoriented) vectors in a temporary array prior to
        interpolation.

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
    cdef void _compute_normalized_weighted_sum_(self, double x, double y, double z, double[::1] &xi):
        """LinearEigenvectorInterpolator._compute_normalized_weighted_sum_(x,
                                                                           y,
                                                                           z,
                                                                           xi
                                                                          )

        A C-level function which performs the local linear interpolation
        within the interpolation voxel, using normalized relative voxel
        coordinates.

        param: x -- Normalized relative voxel coordinates along x-axis
        param: y -- Normalized relative voxel coordinates along y-axis
        param: z -- Normalized relative voxel coordinates along z-axis

        param/return: xi -- Overwritten with the linearly interpolated
                            vector upon exit


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
        dscal(3,c_pow(dnrm2(3,xi,1),-1),xi,1)


    def __dealoc__(self):
        pass


cdef class LinearAimAssister:
    """A Cython extension type tailor-made for computing the projection of a vector
    between two points onto the plane defined as being orthogonal to another
    vector (field).

    Methods defined here:

    LinearAimAssister.__init__(xi_linear)
    LinearAimAssister.set_target(target)
    LinearAimAssister.unset_target()
    LinearAimAssister.__ev__(t,pos)

    Version: 0.2

    """
    cdef:
        double _target_[3]
        double[::1] target
        double _xi_[3]
        double[::1] xi
        double[::1] ret_mv
        LinearEigenvectorInterpolator xi_itp
        bint initialized

    def __cinit__(self):
        self.target = self._target_
        self.xi = self._xi_
        self.ret_mv = np.empty(3)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, LinearEigenvectorInterpolator xi_itp):
        """LinearAimAssister(xi_itp)

        Constructur for the trivariate linear aim assister extension type.

        param: xi_itp -- A LinearEigenvectorInterpolator instance of the (normalized)
                         vector field which locally defines the normal vector
                         to the planes in which one is permitted to aim

        """
        self.xi_itp = xi_itp

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef void set_target(self, double[::1] target):
        """LinearAimAssister.set_target(target)

        Sets the target. Must be called *before* LinearAimAssister.__call__.

        param: target -- Three-component (C-contiguous) NumPy array, containing
                         the Cartesian coordinates of the target.

        """
        if target.shape[0] != 3:
            raise RuntimeError('The interpolation-aiming routine is custom-built' \
                    + ' for three-dimensional data!')

        dcopy(3,target,1,self.target,1)
        self.initialized = True

    cpdef void unset_target(self):
        """LinearAimAssister.unset_target()

        Unsets the target. Useful in order to ensure that the target is always
        updated when it should be.

        """
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos):
        """LinearAimAssister.__call__(t,pos)

        Computes the normalized projection of the vector from 'pos' to 'target'
        (as defined in LinearAimAssister.set_target, which *has* to be called
        in advance) onto the plane whose normal vector at 'pos' is defined by
        a linearly interpolated 'xi' (cf. constructor input)

        param: t   -- Dummy parameter, used by external ODE solver in order to
                      keep track of e.g. arclength (pseudotime parameter)
        param: pos -- Three-component (C-contiguous) NumPy array, containing the
                      Cartesian coordinates at which an aiming direction is
                      sought.

        return: vec -- Three-component (C-contiguous) NumPy array, containing
                       the aforementioned normalized vector projection.

        """
        return np.asarray(self._ev_(pos))

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef double[::1] _ev_(self, double[::1] &pos_):
        """LinearAimAssister._ev_(pos)

        This function facilitates C-level computations with as little Python
        overhead as possible for related extension types (cf. Dp54Linear).

        param: pos -- Three-component (C-contiguous) NumPy array, containing
                      the Cartesian coordinates at which an aiming direction
                      is sought.

        return: vec -- Three-component C-contiguous double memoryview,
                       containing the aforementioned normalized vector
                       projection.

        """
        cdef:
            double[::1] ret_mv = self.ret_mv, xi = self.xi
        if pos_.shape[0] != 3:
            raise RuntimeError('The interpolation-aiming routine is custom-built' \
                    + ' for three dimensional data!')

        if not self.initialized:
            raise RuntimeError('Aim assister not initialized with target!')

        dcopy(3,pos_,1,ret_mv,1)

        xi = self.xi_itp._ev_(ret_mv)

        dscal(3,-1,ret_mv,1)

        daxpy(3,1,self.target,1,ret_mv,1)

        daxpy(3,-ddot(3,xi,1,ret_mv,1),xi,1,ret_mv,1)

        dscal(3,c_pow(dnrm2(3,ret_mv,1),-1),ret_mv,1)

        return ret_mv

    def __dealoc__(self):
        pass



cdef class Dp54Linear:
    """A Cython extension type tailor-made for computing trajectories which
    are defined as being orthogonal to a linearly interpolated trivariate vector
    field in R^3 by means of the Dormand-Prince 5(4) numerical integrator.

    Methods defined here:

    Dp54Linear.__init__(atol, rtol)
    Dp54Linear.set_aim_assister(direction_generator)
    Dp54Linear.unset_aim_assister()

    Version: 0.2

    """
    cdef:
        double fac, maxfac
        double _c_[6]
        double[::1] c
        double _a1_[1]
        double _a2_[2]
        double _a3_[3]
        double _a4_[4]
        double _a5_[5]
        double _a6_[6]
        double[::1] a1, a2, a3, a4, a5, a6
        double _b4_[7]
        double _b5_[7]
        double[::1] b4, b5
        double _x4_[3]
        double _x5_[3]
        double[::1] x4, x5
        double _k1_[3]
        double _k2_[3]
        double _k3_[3]
        double _k4_[3]
        double _k5_[3]
        double _k6_[3]
        double _k7_[3]
        double[::1] k1, k2, k3, k4, k5, k6, k7
        double _k_tmp_[3]
        double[::1] k_tmp
        double _pos_i_[3]
        double[::1] pos_i
        double tmp, sc, err, h_opt
        readonly double atol, rtol
        double q
        LinearAimAssister f
        bint initialized

    def __cinit__(self):
        self._c_  = [       1./5.,     3./10.,    4./5.,      8./9.,             1.,  1.]
        self._a1_ = [       1./5.                                                                      ]
        self._a2_ = [      3./40.,        9./40.                                                       ]
        self._a3_ = [     44./45.,      -56./15.,       32./9.                                         ]
        self._a4_ = [19372./6561., -25360./2187., 64448./6561., -212./729.                             ]
        self._a5_ = [ 9017./3168.,     -355./33., 46732./5247.,   49./176.,  -5103./18656.             ]
        self._a6_ = [    35./384.,            0.,   500./1113.,  125./192.,    -2187./6784.,    11./84.]

        self._b4_ = [5179./57600.,            0., 7571./16695.,  393./640., -92097./339200., 187./2100., 1./40.]
        self._b5_ = [    35./384.,            0.,   500./1113.,  125./192.,    -2187./6784.,    11./84.,     0.]

        self.c = self._c_
        self.a1 = self._a1_
        self.a2 = self._a2_
        self.a3 = self._a3_
        self.a4 = self._a4_
        self.a5 = self._a5_
        self.a6 = self._a6_
        self.b4 = self._b4_
        self.b5 = self._b5_
        self.x4 = self._x4_
        self.x5 = self._x5_

        self.k1 = self._k1_
        self.k2 = self._k2_
        self.k3 = self._k3_
        self.k4 = self._k4_
        self.k5 = self._k5_
        self.k6 = self._k6_
        self.k7 = self._k7_
        self.k_tmp = self._k_tmp_

        self._pos_i_ = np.empty(3)

        self.pos_i = self._pos_i_

        self.fac = 0.8
        self.maxfac = 2.0
        self.q = 4

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double atol, double rtol):
        """Dp54Linear.__init__(atol,rtol)

        Constructor for the trivariate linearly interpolated trajectory
        approximation extension type.

        param: atol -- Absolute tolerance for the Dormand-Prince 5(4) method
        param: rtol -- Relative tolerance for the Dormand-Prince 5(4) method

        """
        self.atol = atol
        self.rtol = rtol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_aim_assister(self, LinearAimAssister direction_generator):
        """Dp54Linear.set_aim_assister(direction_generator)

        Sets the aim assister. Must be called *before* Dp54Linear.__call__.

        param: direction_generator -- LinearAimAssister instance, initialized
                                      with a LinearEigenvectorInterpolator
                                      object as well as a target

        """
        self.f = direction_generator
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def unset_aim_assister(self):
        """Dp54Linear.unset_aim_assister()

        Unsets the aim_assister. Useful in order to ensure that the aim assister
        is always updated when it should be, namely in terms of a new target
        or, for that matter, a new vector field.

        """
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos, double h):
        """Dp54Linear.__call__(t,pos,h)

        Attempts a single step forwards in (pseudo-)time, using the
        Dormand-Prince 5(4) adaptive integrator scheme. If the attempted step
        is rejected, the time level and coordinates are not updated, while the
        time increment is refined.

        Steps in the direction given by the direction generator set by
        set_aim_assister, which has to be called in advance. Otherwise,
        a RuntimeError is raised.

        param: t --   Current (pseudo-)time level
        param: pos -- Current (Cartesian) coordinates, as a three-component
                      C-contiguous NumPy array
        param: h --   Current time increment

        return: t --   (a) New (pseudo-)time level, if the attempted step is
                           accepted
                       (b) Current (pseudo-)time level, if the attempted step is
                           rejected
        return: pos -- Three-component C-contiguous NumPy array, containing
                       (a) Dormand-Prince 5(4) Bspline approximation of the
                           (Cartesian) coordinates at the new time level, if the
                           attempted step is accepted
                       (b) Current (Cartesian) coordinates; unaltered, if the
                           attempted step is rejected
        return: h --   Updated (pseudo-)time increment. Generally increased
                       or decreased, depending on whether or not the trial step
                       is accepted.

        """
        cdef:
            double[::1] pos_i = self.pos_i
        if not self.initialized:
            raise RuntimeError('Dormand-Prince 5(4) linear solver not'\
                    ' initialized with a LinearAimAssister instance!')
        dcopy(3,pos,1,pos_i,1)
        self._ev_(&t,pos_i,&h)
        return t, np.copy(self._pos_i_), h

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _ev_(self, double *t, double[::1] pos, double *h):
        """Dp54Linear._ev_(t,pos,h)

        The C-level function which performs the under-the-hood work of the
        __call__ routine. Facilitates C-level computations with as little
        Python overhead as possible for any related extension types.

        param/return: t
        param/return: pos
        param/return: h

        All params and returns are the same as for the overarching Python-level
        __call__ method.

        """
        cdef:
            double[::1] k1 = self.k1, k2 = self.k2, k3 = self.k3, k4 = self.k4, k5 = self.k5, k6 = self.k6, k7 = self.k7, k_tmp = self.k_tmp
            double[::1] x4 = self.x4, x5 = self.x5
            double[::1] b4 = self.b4, b5 = self.b5
            double[::1] a1 = self.a1, a2 = self.a2, a3 = self.a3, a4 = self.a4, a5 = self.a5, a6 = self.a6
            int i

        k1 = self.f._ev_(pos)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a1[0]*h[0],k1,1,k_tmp,1)

        k2 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a2[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a2[1]*h[0],k2,1,k_tmp,1)

        k3 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a3[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a3[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a3[2]*h[0],k3,1,k_tmp,1)

        k4 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a4[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a4[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a4[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a4[3]*h[0],k4,1,k_tmp,1)

        k5 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a5[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a5[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a5[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a5[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a5[4]*h[0],k5,1,k_tmp,1)

        k6 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a6[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a6[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a6[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a6[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a6[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a6[5]*h[0],k6,1,k_tmp,1)

        k7 = self.f._ev_(k_tmp)

        dscal(3,0,x4,1)
        dscal(3,0,x5,1)

        daxpy(3,b4[0],k1,1,x4,1)
        daxpy(3,b4[1],k2,1,x4,1)
        daxpy(3,b4[2],k3,1,x4,1)
        daxpy(3,b4[3],k4,1,x4,1)
        daxpy(3,b4[4],k5,1,x4,1)
        daxpy(3,b4[5],k6,1,x4,1)
        daxpy(3,b4[6],k7,1,x4,1)

        dscal(3,h[0],x4,1)
        daxpy(3,1,pos,1,x4,1)

        daxpy(3,b5[0],k1,1,x5,1)
        daxpy(3,b5[1],k2,1,x5,1)
        daxpy(3,b5[2],k3,1,x5,1)
        daxpy(3,b5[3],k4,1,x5,1)
        daxpy(3,b5[4],k5,1,x5,1)
        daxpy(3,b5[5],k6,1,x5,1)
        daxpy(3,b5[6],k7,1,x5,1)

        dscal(3,h[0],x5,1)
        daxpy(3,1,pos,1,x5,1)

        if c_fabs(x4[0]) < c_fabs(x5[0]):
            self.sc = self.atol + c_fabs(x4[0])*self.rtol
        else:
            self.sc = self.atol + c_fabs(x5[0])*self.rtol

        self.err = c_fabs(x4[0]-x5[0])/self.sc
        for i in range(1,3):
            if c_fabs(x4[i]) < c_fabs(x5[i]):
                self.sc = self.atol + c_fabs(x4[i])*self.rtol
            else:
                self.sc = self.atol + c_fabs(x5[i])*self.rtol

            self.tmp = c_fabs(x4[i]-x5[i])/self.sc

            if self.tmp > self.err:
                self.err = self.tmp

        if self.err == 0:
            self.h_opt = c_copysign(10,h[0])
        else:
            self.h_opt = h[0]*c_pow((1/self.err),1/(self.q+1))

        if self.err < 1:
            t[0] = t[0] + h[0]
            dcopy(3,x5,1,pos,1)
            if self.maxfac*h[0] < self.fac*self.h_opt:
                h[0] = self.maxfac*h[0]
            else:
                h[0] = self.fac*self.h_opt
        else:
            h[0] = self.fac*self.h_opt


cdef class Rk4Linear:
    """A Cython extension type tailor-made for computing trajectories which
    are defined as being orthogonal to a linearly interpolated trivariate vector
    field in R^3 by means of the RK4 numerical integrator.

    Methods defined here:

    Rk4Linear.__init__(atol, rtol)
    Rk4Linear.set_aim_assister(direction_generator)
    Rk54Linear.unset_aim_assister()

    Version: 0.2

    """
    cdef:
        double _c_[4]
        double[::1] c
        double _a1_[1]
        double _a2_[2]
        double _a3_[3]
        double[::1] a1, a2, a3
        double _k1_[3]
        double _k2_[3]
        double _k3_[3]
        double _k4_[3]
        double[::1] k1, k2, k3, k4
        double _k_tmp_[3]
        double[::1] k_tmp
        double _b_[4]
        double[::1] b
        double _pos_i_[3]
        double[::1] pos_i
        LinearAimAssister f
        bint initialized

    def __cinit__(self):
        self._c_  = [0.   ,  0.5,   0.5,   1    ]
        self._a1_ = [0.5                        ]
        self._a2_ = [0.   ,  0.5                ]
        self._a3_ = [0.   ,  0.,    1.          ]

        self._b_ =  [1./6.,  1./3., 1./3., 1./6.]

        self.c = self._c_
        self.a1 = self._a1_
        self.a2 = self._a2_
        self.a3 = self._a3_

        self.k1 = self._k1_
        self.k2 = self._k2_
        self.k3 = self._k3_
        self.k4 = self._k4_

        self.k_tmp = self._k_tmp_

        self.b = self._b_

        self._pos_i_ = np.empty(3)

        self.pos_i = self._pos_i_

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self):
        """Rk54Linear.__init__()

        Constructor for the trivariate linearly interpolated trajectory
        approximation extension type.


        """

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_aim_assister(self, LinearAimAssister direction_generator):
        """Rk54Linear.set_aim_assister(direction_generator)

        Sets the aim assister. Must be called *before* Dp54Linear.__call__.

        param: direction_generator -- LinearAimAssister instance, initialized
                                      with a LinearEigenvectorInterpolator
                                      object as well as a target

        """
        self.f = direction_generator
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def unset_aim_assister(self):
        """Rk4Linear.unset_aim_assister()

        Unsets the aim_assister. Useful in order to ensure that the aim assister
        is always updated when it should be, namely in terms of a new target
        or, for that matter, a new vector field.

        """
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos, double h):
        """Rk4Linear.__call__(t,pos,h)

        Attempts a single step forwards in (pseudo-)time, using the
        RK4 numerical integrator scheme. If the attempted step
        is rejected, the time level and coordinates are not updated, while the
        time increment is refined.

        Steps in the direction given by the direction generator set by
        set_aim_assister, which has to be called in advance. Otherwise,
        a RuntimeError is raised.

        param: t --   Current (pseudo-)time level
        param: pos -- Current (Cartesian) coordinates, as a three-component
                      C-contiguous NumPy array
        param: h --   Current time increment

        return: t --   New (pseudo-)time level
        return: pos -- Three-component C-contiguous NumPy array, containing
                       RK4 linear approximation of the (Cartesian) coordinates
                       at the new time level, if the attempted step is accepted
        return: h --   (Pseudo-)time increment. Unaltered.

        """
        cdef:
            double[::1] pos_i = self.pos_i
        if not self.initialized:
            raise RuntimeError('RK4 linear solver not'\
                    ' initialized with a LinearAimAssister instance!')
        dcopy(3,pos,1,pos_i,1)
        self._ev_(&t,pos_i,&h)
        return t, np.copy(pos_i), h

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _ev_(self, double *t, double[::1] pos, double *h):
        """Rk4Linear._ev_(t,pos,h)

        The C-level function which performs the under-the-hood work of the
        __call__ routine. Facilitates C-level computations with as little
        Python overhead as possible for any related extension types.

        param/return: t
        param/return: pos
        param/return: h

        All params and returns are the same as for the overarching Python-level
        __call__ method.

        """
        cdef:
            double[::1] k1 = self.k1, k2 = self.k2, k3 = self.k3, k4 = self.k4
            double[::1] a1 = self.a1, a2 = self.a2, a3 = self.a3
            double[::1] b = self.b
            double[::1] k_tmp = self.k_tmp

        k1 = self.f._ev_(pos)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a1[0]*h[0],k1,1,k_tmp,1)

        k2 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a2[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a2[1]*h[0],k2,1,k_tmp,1)

        k3 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a3[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a3[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a3[2]*h[0],k3,1,k_tmp,1)

        k4 = self.f._ev_(k_tmp)

        dscal(3,0,k_tmp,1)

        daxpy(3,b[0],k1,1,k_tmp,1)
        daxpy(3,b[1],k2,1,k_tmp,1)
        daxpy(3,b[2],k3,1,k_tmp,1)
        daxpy(3,b[3],k4,1,k_tmp,1)

        dscal(3,h[0],k_tmp,1)

        daxpy(3,1,k_tmp,1,pos,1)

        t[0] += h[0]

cdef class Bs32Linear:
    """A Cython extension type tailor-made for computing trajectories which
    are defined as being orthogonal to a linearly interpolated trivariate vector
    field in R^3 by means of the Dormand-Prince 5(4) numerical integrator.

    Methods defined here:

    Bs32Linear.__init__(atol, rtol)
    Bs32Linear.set_aim_assister(direction_generator)
    Bs32Linear.unset_aim_assister()

    Version: 0.2

    """
    cdef:
        double fac, maxfac
        double _c_[3]
        double[::1] c
        double _a1_[1]
        double _a2_[2]
        double _a3_[3]
        double[::1] a1, a2, a3
        double _b2_[4]
        double _b3_[4]
        double[::1] b2, b3
        double _x2_[3]
        double _x3_[3]
        double[::1] x2, x3
        double _k1_[3]
        double _k2_[3]
        double _k3_[3]
        double _k4_[3]
        double[::1] k1, k2, k3, k4
        double _k_tmp_[3]
        double[::1] k_tmp
        double _pos_i_[3]
        double[::1] pos_i
        double tmp, sc, err, h_opt
        readonly double atol, rtol
        double q
        LinearAimAssister f
        bint initialized

    def __cinit__(self):
        self._c_  = [ 1./2., 3./4.,    1.       ]
        self._a1_ = [ 1./2.                     ]
        self._a2_ = [    0., 3./4.              ]
        self._a3_ = [ 2./9., 1./3., 4./9.       ]

        self._b3_ = [ 2./9., 1./3., 4./9.,    0.]
        self._b2_ = [7./24., 1./4., 1./3., 1./8.]

        self.c = self._c_
        self.a1 = self._a1_
        self.a2 = self._a2_
        self.a3 = self._a3_
        self.b3 = self._b3_
        self.b2 = self._b2_
        self.x3 = self._x3_
        self.x2 = self._x2_

        self.k1 = self._k1_
        self.k2 = self._k2_
        self.k3 = self._k3_
        self.k4 = self._k4_
        self.k_tmp = self._k_tmp_

        self._pos_i_ = np.empty(3)

        self.pos_i = self._pos_i_

        self.fac = 0.8
        self.maxfac = 2.0
        self.q = 2

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double atol, double rtol):
        """Bs32Linear.__init__(atol,rtol)

        Constructor for the trivariate linearly interpolated trajectory
        approximation extension type.

        param: atol -- Absolute tolerance for the Dormand-Prince 5(4) method
        param: rtol -- Relative tolerance for the Dormand-Prince 5(4) method

        """
        self.atol = atol
        self.rtol = rtol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_aim_assister(self, LinearAimAssister direction_generator):
        """Bs32Linear.set_aim_assister(direction_generator)

        Sets the aim assister. Must be called *before* Bs32Linear.__call__.

        param: direction_generator -- LinearAimAssister instance, initialized
                                      with a LinearEigenvectorInterpolator
                                      object as well as a target

        """
        self.f = direction_generator
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def unset_aim_assister(self):
        """Bs32Linear.unset_aim_assister()

        Unsets the aim_assister. Useful in order to ensure that the aim assister
        is always updated when it should be, namely in terms of a new target
        or, for that matter, a new vector field.

        """
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos, double h):
        """Bs32Linear.__call__(t,pos,h)

        Attempts a single step forwards in (pseudo-)time, using the
        Dormand-Prince 5(4) adaptive integrator scheme. If the attempted step
        is rejected, the time level and coordinates are not updated, while the
        time increment is refined.

        Steps in the direction given by the direction generator set by
        set_aim_assister, which has to be called in advance. Otherwise,
        a RuntimeError is raised.

        param: t --   Current (pseudo-)time level
        param: pos -- Current (Cartesian) coordinates, as a three-component
                      C-contiguous NumPy array
        param: h --   Current time increment

        return: t --   (a) New (pseudo-)time level, if the attempted step is
                           accepted
                       (b) Current (pseudo-)time level, if the attempted step is
                           rejected
        return: pos -- Three-component C-contiguous NumPy array, containing
                       (a) Dormand-Prince 5(4) Bspline approximation of the
                           (Cartesian) coordinates at the new time level, if the
                           attempted step is accepted
                       (b) Current (Cartesian) coordinates; unaltered, if the
                           attempted step is rejected
        return: h --   Updated (pseudo-)time increment. Generally increased
                       or decreased, depending on whether or not the trial step
                       is accepted.

        """
        cdef:
            double[::1] pos_i = self.pos_i
        if not self.initialized:
            raise RuntimeError('Dormand-Prince 5(4) linear solver not'\
                    ' initialized with a LinearAimAssister instance!')
        dcopy(3,pos,1,pos_i,1)
        self._ev_(&t,pos_i,&h)
        return t, np.copy(self._pos_i_), h

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _ev_(self, double *t, double[::1] pos, double *h):
        """Bs32Linear._ev_(t,pos,h)

        The C-level function which performs the under-the-hood work of the
        __call__ routine. Facilitates C-level computations with as little
        Python overhead as possible for any related extension types.

        param/return: t
        param/return: pos
        param/return: h

        All params and returns are the same as for the overarching Python-level
        __call__ method.

        """
        cdef:
            double[::1] k1 = self.k1, k2 = self.k2, k3 = self.k3, k4 = self.k4, k_tmp = self.k_tmp
            double[::1] x2 = self.x2, x3 = self.x3
            double[::1] b3 = self.b3, b2 = self.b2
            double[::1] a1 = self.a1, a2 = self.a2, a3 = self.a3
            int i

        k1 = self.f._ev_(pos)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a1[0]*h[0],k1,1,k_tmp,1)

        k2 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a2[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a2[1]*h[0],k2,1,k_tmp,1)

        k3 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a3[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a3[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a3[2]*h[0],k3,1,k_tmp,1)

        k4 = self.f._ev_(k_tmp)

        dscal(3,0,x3,1)
        dscal(3,0,x2,1)

        daxpy(3,b2[0],k1,1,x2,1)
        daxpy(3,b2[1],k2,1,x2,1)
        daxpy(3,b2[2],k3,1,x2,1)
        daxpy(3,b2[3],k4,1,x2,1)

        dscal(3,h[0],x2,1)
        daxpy(3,1,pos,1,x2,1)

        daxpy(3,b3[0],k1,1,x3,1)
        daxpy(3,b3[1],k2,1,x3,1)
        daxpy(3,b3[2],k3,1,x3,1)
        daxpy(3,b3[3],k4,1,x3,1)

        dscal(3,h[0],x3,1)
        daxpy(3,1,pos,1,x3,1)

        if c_fabs(x2[0]) < c_fabs(x3[0]):
            self.sc = self.atol + c_fabs(x2[0])*self.rtol
        else:
            self.sc = self.atol + c_fabs(x3[0])*self.rtol

        self.err = c_fabs(x2[0]-x3[0])/self.sc
        for i in range(1,3):
            if c_fabs(x2[i]) < c_fabs(x3[i]):
                self.sc = self.atol + c_fabs(x2[i])*self.rtol
            else:
                self.sc = self.atol + c_fabs(x3[i])*self.rtol

            self.tmp = c_fabs(x2[i]-x3[i])/self.sc

            if self.tmp > self.err:
                self.err = self.tmp

        if self.err == 0:
            self.h_opt = c_copysign(10,h[0])
        else:
            self.h_opt = h[0]*c_pow((1/self.err),1/(self.q+1))

        if self.err < 1:
            t[0] = t[0] + h[0]
            dcopy(3,x3,1,pos,1)
            if self.maxfac*h[0] < self.fac*self.h_opt:
                h[0] = self.maxfac*h[0]
            else:
                h[0] = self.fac*self.h_opt
        else:
            h[0] = self.fac*self.h_opt

cdef class Dp87Linear:
    """A Cython extension type tailor-made for computing trajectories which
    are defined as being orthogonal to a linearly interpolated trivariate vector
    field in R^3 by means of the Dormand-Prince 8(7) numerical integrator.

    Methods defined here:

    Dp87Linear.__init__(atol, rtol)
    Dp87Linear.set_aim_assister(direction_generator)
    Dp87Linear.unset_aim_assister()

    Version: 0.2

    """
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
        readonly double atol, rtol
        double q
        LinearAimAssister f
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

        self._pos_i_ = np.empty(3)

        self.pos_i = self._pos_i_

        self.fac = 0.8
        self.maxfac = 2.0
        self.q = 4

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double atol, double rtol):
        """Dp87Linear.__init__(atol,rtol)

        Constructor for the trivariate linearly interpolated trajectory
        approximation extension type.

        param: atol -- Absolute tolerance for the Dormand-Prince 8(7) method
        param: rtol -- Relative tolerance for the Dormand-Prince 8(7) method

        """
        self.atol = atol
        self.rtol = rtol



    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_aim_assister(self, LinearAimAssister direction_generator):
        """Dp87Linear.set_aim_assister(direction_generator)

        Sets the aim assister. Must be called *before* Dp87Linear.__call__.

        param: direction_generator -- LinearAimAssister instance, initialized
                                      with a LinearEigenvectorInterpolator
                                      object as well as a target

        """
        self.f = direction_generator
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def unset_aim_assister(self):
        """Dp87Linear.unset_aim_assister()

        Unsets the aim_assister. Useful in order to ensure that the aim assister
        is always updated when it should be, namely in terms of a new target
        or, for that matter, a new vector field.

        """
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos, double h):
        """Dp87Linear.__call__(t,pos,h)

        Attempts a single step forwards in (pseudo-)time, using the
        Dormand-Prince 8(7) adaptive integrator scheme. If the attempted step
        is rejected, the time level and coordinates are not updated, while the
        time increment is refined.

        Steps in the direction given by the direction generator set by
        set_aim_assister, which has to be called in advance. Otherwise,
        a RuntimeError is raised.

        param: t --   Current (pseudo-)time level
        param: pos -- Current (Cartesian) coordinates, as a three-component
                      C-contiguous NumPy array
        param: h --   Current time increment

        return: t --   (a) New (pseudo-)time level, if the attempted step is
                           accepted
                       (b) Current (pseudo-)time level, if the attempted step is
                           rejected
        return: pos -- Three-component C-contiguous NumPy array, containing
                       (a) Dormand-Prince 8(7) Bspline approximation of the
                           (Cartesian) coordinates at the new time level, if the
                           attempted step is accepted
                       (b) Current (Cartesian) coordinates; unaltered, if the
                           attempted step is rejected
        return: h --   Updated (pseudo-)time increment. Generally increased
                       or decreased, depending on whether or not the trial step
                       is accepted.

        """
        cdef:
            double[::1] pos_i = self.pos_i
        if not self.initialized:
            raise RuntimeError('Dormand-Prince 8(7) linear solver not'\
                    ' initialized with a LinearAimAssister instance!')
        dcopy(3,pos,1,pos_i,1)
        self._ev_(&t,pos_i,&h)
        return t, np.copy(self._pos_i_), h

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _ev_(self, double *t, double[::1] pos, double *h):
        """Dp87Linear._ev_(t,pos,h)

        The C-level function which performs the under-the-hood work of the
        __call__ routine. Facilitates C-level computations with as little
        Python overhead as possible for any related extension types.

        param/return: t
        param/return: pos
        param/return: h

        All params and returns are the same as for the overarching Python-level
        __call__ method.

        """
        cdef:
            double[::1] k1 = self.k1, k2 = self.k2, k3 = self.k3, k4 = self.k4, k5 = self.k5, k6 = self.k6, k7 = self.k7, k8 = self.k8, k9 = self.k9, k10 = self.k10, k11 = self.k11, k12 = self.k12, k13 = self.k13, k_tmp = self.k_tmp
            double[::1] x7 = self.x7, x8 = self.x8
            double[::1] b7 = self.b7, b8 = self.b8
            double[::1] a1 = self.a1, a2 = self.a2, a3 = self.a3, a4 = self.a4, a5 = self.a5, a6 = self.a6, a7 = self.a7, a8 = self.a8, a9 = self.a9, a10 = self.a10, a11 = self.a11, a12 = self.a12, a13 = self.a13
            int i

        k1 = self.f._ev_(pos)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a1[0]*h[0],k1,1,k_tmp,1)

        k2 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a2[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a2[1]*h[0],k2,1,k_tmp,1)

        k3 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a3[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a3[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a3[2]*h[0],k3,1,k_tmp,1)

        k4 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a4[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a4[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a4[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a4[3]*h[0],k4,1,k_tmp,1)

        k5 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a5[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a5[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a5[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a5[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a5[4]*h[0],k5,1,k_tmp,1)

        k6 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a6[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a6[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a6[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a6[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a6[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a6[5]*h[0],k6,1,k_tmp,1)

        k7 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a7[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a7[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a7[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a7[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a7[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a7[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a7[6]*h[0],k7,1,k_tmp,1)

        k8 = self.f._ev_(k_tmp)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a8[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a8[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a8[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a8[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a8[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a8[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a8[6]*h[0],k7,1,k_tmp,1)
        daxpy(3,a8[7]*h[0],k8,1,k_tmp,1)

        k9 = self.f._ev_(k_tmp)
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

        k10 = self.f._ev_(k_tmp)
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

        k11 = self.f._ev_(k_tmp)
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

        k12 = self.f._ev_(k_tmp)
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

        k13 = self.f._ev_(k_tmp)

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
