# distutils: language = c++

"""This module contains a Cython extension type which facilitates B-spline
interpolation of trivariate vector fields in R^3, making use of the
Bspline-Fortran library, which is available at
    https://github.com/jacobwilliams/bspline-fortran

Furthermore, a Cython extension type which computes a unit normalized orthogonal
projection of a vector between two points into the surface which locally can be
approximated as a plane orthogonal to a trivariate vector field in R^3.

Lastly, a Cython extension type which arranges for Dormand-Prince 5(4)
approximation of trajectories which are defined as being orthogonal
to a trivariate vector field in R^3

Extension types defined here:
    SplineEigenvectorInterpolator
    SplineAimAssister
    Dp54BSpline

Written by Arne Magnus T. Løken as part of my master's thesis work in physics
at NTNU, spring 2018.

"""
cimport numpy as np
import numpy as np

cimport cython

from scipy.linalg.cython_blas cimport ddot as scp_ddot, dnrm2 as scp_dnrm2,\
                                      daxpy as scp_daxpy, dscal as scp_dscal,\
                                      dcopy as scp_dcopy

from libc.math cimport pow as c_pow, fmod as c_fmod, floor as c_floor, \
                       copysign as c_copysign, fabs as c_fabs

from libcpp.vector cimport vector

@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef double ddot(int N, double[::1] &x, int INCX, double[::1] &y, int INCY) nogil:
    return scp_ddot(&N, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void dcopy(int N, double[::1] &x, int INCX, double[::1] &y, int INCY) nogil:
    scp_dcopy(&N, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double dnrm2(int N, double[::1] &x, int INCX) nogil:
    return scp_dnrm2(&N, &x[0], &INCX)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void daxpy(int N, double alpha, double[::1] &x, int INCX, double[::1] &y, int INCY) nogil:
    scp_daxpy(&N, &alpha, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void dscal(int N, double alpha, double[::1] &x, int INCX) nogil:
    scp_dscal(&N, &alpha, &x[0], &INCX)

cdef extern from "linkto.h":
    cppclass ItpCont:
        ItpCont() except +
        void init_interp(double *x, double *y, double *z, double *f,
                int kx, int ky, int kz, int nx, int ny, int nz, int ext) except +
        double eval_interp(double x, double y, double z, int dx, int dy, int dz) except +
        void kill_interp() except +

ctypedef vector[ItpCont*] ItpCont_ptr_vec

cdef class SplineEigenvectorInterpolator:
    """A Cython extension type tailor-made for computing a special-purpose
    (higher order) spline interpolation of a three-dimensional, trivariate
    vector field, subject to periodic boundary conditions, with a local
    orientation fix prior to interpolation in order to combat local
    orientational discontinuities which may arise due to numerical noise.

    Methods defined here:
    SplineEigenvectorInterpolator.__init__(x,y,z,xi,kx,ky,kz)
    SplineEigenvectorInterpolator.__call__(pos)

    Version: 0.2

    """
    cdef:
        ItpCont_ptr_vec interp
        double[:,:,:,::1] xi
        double[::1] x, y, z
        double dx, dy, dz
        double _x_[4]
        double _y_[4]
        double _z_[4]
        double[::1] x_, y_, z_
        int _ixs_[4]
        int _iys_[4]
        int _izs_[4]
        int[::1] ixs, iys, izs
        double _xivals_[3][64]
        double[:,::1] xivals
        int nx, ny, nz
        double _xival_[3]
        double[::1] xival_
        double _xia_[3]
        double[::1] xia_
        double _xiref_[3]
        double[::1] xiref_
        double _pos_internal_[3]
        double[::1] pos_
        double[::1] ret_mv
        bint calibrated, sameaslast
        int kx, ky, kz

    def __cinit__(self):
        self.interp = ItpCont_ptr_vec(3)
        self.interp[0] = new ItpCont()
        self.interp[1] = new ItpCont()
        self.interp[2] = new ItpCont()
        self.x_ = self._x_
        self.y_ = self._y_
        self.z_ = self._z_
        self.ixs = self._ixs_
        self.iys = self._iys_
        self.izs = self._izs_
        self.xivals = self._xivals_
        self.xival_ = self._xival_
        self.xia_ = self._xia_
        self.xiref_ = self._xiref_
        self.pos_ = self._pos_internal_
        self.calibrated = False
        self.ret_mv = np.empty(3)
        self.sameaslast = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, double[::1] x not None, double[::1] y not None, double[::1] z not None, \
            double[:,:,:,::1] xi not None, int kx = 3, int ky = 3, int kz = 3):
        """SplineEigenvectorInterpolator.__init__(x,y,z,xi,kx,ky,kz)

        Constructor for the three-dimensional, trivariate B-spline interpolator
        extension type. Periodic boundary conditions are enforced.

        param: x  -- Sampling points along the x abscissa, as a (C-contiguous)
                     NumPy array of shape (nx >= 4) and type np.float64
        param: y  -- Sampling points along the y abscissa, as a (C-contiguous)
                     NumPy array of shape (ny >= 4) and type np.float64
        param: z  -- Sampling points along the z abscissa, as a (C-contiguous)
                     NumPy array of shape (nz >= 4) and type np.float64
        param: xi -- Sampled vector field at the grid spanned by the input
                     arrays x, y and z, as a (C-contiguous) NumPy array of
                     shape (nx,ny,nz,3) and type np.float64
        OPTIONAL:
        param: kx -- Interpolation order along the x-axis. Possible choices:
                     kx = 2, kx = 3. DEFAULT: kx = 3.
        param: ky -- Interpolation order along the y-axis. Possible choices:
                     ky = 2, ky = 3. DEFAULT: ky = 3.
        param: kz -- Interpolation order along the z-axis. Possible choices:
                     kz = 2, kz = 3. DEFAULT: kz = 3.

        """
        cdef:
            int i
        if(xi.shape[0] != x.shape[0] or xi.shape[1] != y.shape[0] or xi.shape[2] != z.shape[0]):
            raise RuntimeError('Array dimensions not aligned!')

        if(xi.shape[3] != 3):
            raise RuntimeError('The interpolator routine is custom-built for three dimensional data!')

        if (kx < 2 or kx > 3) or (ky < 2 or ky > 3) or (kz < 2 or kz > 3):
            raise RuntimeError('')
        # Enforcing periodic BC by not including the sampling points along
        # the last rows and columns
        self.nx = x.shape[0]-1
        self.ny = y.shape[0]-1
        self.nz = z.shape[0]-1

        self.x = x[:self.nx]
        self.y = y[:self.ny]
        self.z = z[:self.nz]

        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dz = z[1]-z[0]

        # Set local coordinates for interpolation within any given 4x4x4 voxel
        for i in range(4):
            self.x_[i] = (i-1)*self.dx
            self.y_[i] = (i-1)*self.dy
            self.z_[i] = (i-1)*self.dz

        self.xi = xi[:self.nx,:self.ny,:self.nz,:]

        self.kx = kx
        self.ky = ky
        self.kz = kz

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double[::1] pos not None):
        """SplineEigenvectorInterpolator.__call__(pos)

        Computes a spline interpolation of the vector field at 'pos'
        based upon the 4x4x4 set of nearest neighbor voxels, including
        a local direction fix --- i.e., ensuring that no pair of vectors
        is rotated more than 90 degrees with respect to eachother.

        Periodic boundary conditions are built in.

        param: pos -- Three-component (C-contiguous) NumPy array,
                      containing the Cartesian coordinates at which a spline
                      interpolated vector is sought

        return: vec -- Three-component (C-contiguous) NumPy array, containing
                       the aforementioned, normalized vector

        """
        return np.copy(self._ev_(pos))

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef double[::1] _ev_(self, double[::1] pos_):
        """SplineEigenvectorInterpolator._ev_(pos)

        The C-level function which computes the spline interpolated vector
        field, which is returned by the __call__ routine.

        This function facilitates C-level computations with as little Python
        overhead as possible for related extension types (cf. SplineAimAssister,
        Dp54BSpline).

        param: pos_ -- Three-component C-contiguous memoryview of doubles,
                       containing the Cartesian coordinates at which a spline
                       interpolated vector is sought

        return: vec -- Three-component C-contiguous memoryview of doubles,
                       containing the normalized Cartesian coordinates of the
                       spline interpolated eigenvector.

        """
        cdef:
            double[::1] ret_mv = self.ret_mv, xi = self.xival_
            double[::1] pos = self.pos_

        if pos_.shape[0] != 3:
            raise RuntimeError('The interpolation routine is custom-built'\
                    +' for three dimensional data')

        dcopy(3,pos_,1,pos,1)

        self._interpolate_xi_(pos[0],pos[1],pos[2],xi)
        dcopy(3,xi,1,ret_mv,1)

        return ret_mv

    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _interpolate_xi_(self, double x, double y, double z, double[::1] &xi):
        """SplineEigenvectorInterpolator._interpolate_xi_(x,y,z,xi)

        The C-level function which computes the spline interpolated vector
        field, which is returned by the __call__ routine.

        This function facilitates C-level computations with as little Python
        overhead as possible for related extension types (cf. SplineAimAssister,
        Dp54BSpline).

        param: x -- x-coordinate at which an interpolated vector is sought
        param: y -- y-coordinate at which an interpolated vector is sought
        param: z -- z-coordinate at which an interpolated vector is sought

        param/return: xi -- Three-component C-contiguous double memoryview,
                            which upon exit is overwritten with the interpolated
                            eigenvector.

        """
        cdef:
            int i, j, k,l
            int[::1] ixs = self.ixs, iys = self.iys, izs = self.izs
            double[::1] x_ = self.x_, y_ = self.y_, z_ = self.z_
            double[:,::1] xivals = self.xivals
            double[:,:,:,::1] xis = self.xi
            double[::1] xia = self.xia_, xiref = self.xiref_
        x = c_fmod(x,self.x[self.nx-1])
        y = c_fmod(y,self.y[self.ny-1])
        z = c_fmod(z,self.z[self.nz-1])

        while x < 0:
            x += self.x[self.nx-1]
        while y < 0:
            y += self.y[self.ny-1]
        while z < 0:
            z += self.z[self.nz-1]
        self._set_voxel_indices_(x,y,z)
        x -= self.x[ixs[1]]
        y -= self.y[iys[1]]
        z -= self.z[izs[1]]
        if not (self.sameaslast and self.calibrated):
            dcopy(3,xis[ixs[0],iys[0],izs[0]],1,xiref,1)
            for k in range(4):
                for j in range(4):
                    for i in range(4):
                        dcopy(3,xis[ixs[i],iys[j],izs[k]],1,xia,1)
                        if ddot(3,xia,1,xiref,1) < 0:
                            for l in range(3):
                                xivals[l,i+4*(j+4*k)] = -xia[l]
                        else:
                            for l in range(3):
                                xivals[l,i+4*(j+4*k)] = xia[l]

            for l in range(3):
                self.interp[l].init_interp(&x_[0],&y_[0],&z_[0],&xivals[l,0],self.kx,self.ky,self.kz,4,4,4,0)
            self.calibrated = True
        for l in range(3):
            xi[l] = self.interp[l].eval_interp(x,y,z,0,0,0)
        dscal(3,c_pow(dnrm2(3,xi,1),-1),xi,1)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.initializedcheck(False)
    cdef void _set_voxel_indices_(self,double x, double y, double z):
        """SplineEigenvectorInterpolator._set_voxel_indices_(x,y,z,xi)

        A C-level function which computes the indices of the points within
        the interpolation voxel which surrounds the point specified by the
        Cartesian coordinates [x,y,z].

        param: x -- x-coordinate at which an interpolated vector is sought
        param: y -- y-coordinate at which an interpolated vector is sought
        param: z -- z-coordinate at which an interpolated vector is sought

        """
        cdef:
            int[::1] ixs = self.ixs, iys = self.iys, izs = self.izs
            int ix, iy, iz

        x = c_fmod((x-self.x[0])/(self.dx),self.nx-1)
        y = c_fmod((y-self.y[0])/(self.dy),self.ny-1)
        z = c_fmod((z-self.z[0])/(self.dz),self.nz-1)

        while x < 0:
            x += self.nx-1
        while y < 0:
            y += self.ny-1
        while z < 0:
            z += self.nz-1

        ix = int(c_floor(x))
        iy = int(c_floor(y))
        iz = int(c_floor(z))

        if not self.calibrated or (self.ixs[1] != ix or self.iys[1] != iy
                                    or self.izs[1] != iz):

            ixs[1] = ix
            iys[1] = iy
            izs[1] = iz

            ixs[0] = (ix-1)%(self.nx)
            ixs[2] = (ix+1)%(self.nx)
            ixs[3] = (ix+2)%(self.nx)

            while ixs[0] < 0:
                ixs[0] += self.nx
            while ixs[2] < 0:
                ixs[2] += self.nx
            while ixs[3] < 0:
                ixs[3] += self.nx

            iys[0] = (iy-1)%(self.ny)
            iys[2] = (iy+1)%(self.ny)
            iys[3] = (iy+2)%(self.ny)

            while iys[0] < 0:
                iys[0] += self.ny
            while iys[2] < 0:
                iys[2] += self.ny
            while iys[3] < 0:
                iys[3] += self.ny

            izs[0] = (iz-1)%(self.nz)
            izs[2] = (iz+1)%(self.nz)
            izs[3] = (iz+2)%(self.nz)

            while izs[0] < 0:
                izs[0] += self.nz
            while izs[2] < 0:
                izs[2] += self.nz
            while izs[3] < 0:
                izs[3] += self.nz

            self.sameaslast = False
        else:
            self.sameaslast = True

    def __dealoc__(self):
        if self.interp[0] is not NULL:
            del self.interp[0]
        if self.interp[1] is not NULL:
            del self.interp[1]
        if self.interp[2] is not NULL:
            del self.interp[2]

cdef class SplineAimAssister:
    """A Cython extension type tailor-made for computing the projection of
    a vector between two points onto the plane defined as being orthogonal to
    another vector (field).

    Methods defined here:

    SplineAimAssister.__init__(xi_splined)
    SplineAimAssister.set_target(target)
    SplineAimAssister.unset_target()
    SplineAimAssister.__ev__(t,pos)

    Version: 0.2

    """
    cdef:
        double _target_[3]
        double[::1] target
        double _xi_[3]
        double[::1] xi
        double _pos_[3]
        double[::1] pos
        double[::1] ret_mv
        SplineEigenvectorInterpolator xi_itp
        bint initialized

    def __cinit__(self):
        self.target = self._target_
        self.xi = self._xi_
        self.pos = self._pos_
        self.ret_mv = np.empty(3)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, SplineEigenvectorInterpolator xi_itp):
        """SplineAimAssister(xi_itp)

        Constructor for the trivariate B-spline aim assister extension type.

        param: xi_itp -- A SplineEigenInterpolator instance of the (normalized)
                         vector field which locally defines the normal vector
                         to the planes in which one is permitted to aim.

        """
        self.xi_itp = xi_itp

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef void set_target(self, double[::1] target):
        """SplineAimAssister.set_target(target)

        Sets the target. Must be called *before* SplineAimAssister.__call__.

        param: target -- Three-component (C-contiguous) NumPy array,

                         containing the Cartesian coordinates of the target.
        """
        if target.shape[0] != 3:
            raise RuntimeError('The interpolation-aiming routine is custom-built'\
                    +' for three dimensional data!')
        dcopy(3,target,1,self.target,1)
        self.initialized = True

    cpdef void unset_target(self):
        """SplineAimAssister.unset_target()

        Unsets the target. Useful in order to ensure that the target is always
        updated when it should be.

        """
        self.initialized = False


    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos):
        """SplineAimAssister.__call__(t,pos)

        Computes the normalized projection of the vector from 'pos' to 'target'
        (as defined in SplineAimAssister.set_target, which *has* to be called
        in advance) onto the plane whose normal vector at 'pos' is defined
        by a B-spline interpolated 'xi' (cf. constructor input)

        param: t   -- Dummy parameter, used by external ODE solver in order to
                      keep track of e.g. arclength (pseudotime parameter)
        param: pos -- Three-component (C-contiguous) NumPy array,
                      containing the Cartesian coordinates at which an aiming
                      direction is sought.

        return: vec -- Three-component (C-contiguous) NumPy array, containing
                       the aforementioned normalized vector projection.

        """
        return np.copy(self._ev_(pos))

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef double[::1] _ev_(self, double[::1] &pos_):
        """SplineAimAssister._ev_(pos)

        This function facilitates C-level computations with as little Python
        overhead as possible for related extension types (cf. Dp54BSpline).

        param: pos -- Three-component (C-contiguous) NumPy array, containing
                      the Cartesian coordinates at which an aiming direction
                      is sought.

        return: vec -- Three-component C-contiguous double memoryview,
                       containing the aforementioned normalized vector
                       projection.

        """
        cdef:
            double[::1] ret_mv = self.ret_mv, xi = self.xi
            double[::1] pos = self.pos
        if pos_.shape[0] != 3:
            raise RuntimeError('The interpolation-aiming routine is custom-built'\
                    +' for three dimensional data!')

        if not self.initialized:
            raise RuntimeError('Aim assister not initialized with target')

        dcopy(3,pos_,1,pos,1)

        xi = self.xi_itp._ev_(pos)

        ret_mv = pos


        dscal(3,-1.,ret_mv,1)

        daxpy(3,1.,self.target,1,ret_mv,1)
        daxpy(3,-ddot(3,xi,1,ret_mv,1),xi,1,ret_mv,1)


        # In order to combat NANs:
        while dnrm2(3,ret_mv,1) < 0.001:
            dscal(3,100,ret_mv,1)

        dscal(3,c_pow(dnrm2(3,ret_mv,1),-1),ret_mv,1)


        return ret_mv


    def __dealoc__(self):
        pass

cdef class Dp54BSpline:
    """A Cython extension type tailor-made for computing trajectories which
    are defined as being orthogonal to a B-spline interpolated trivariate vector
    field in R^3 by means of the Dormand-Prince 5(4) numerical integrator.

    Methods defined here:

    Dp54BSpline.__init__(atol, rtol)
    Dp54BSpline.set_aim_assister(direction_generator)
    Dp54BSpline.unset_aim_assister()

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
        double _x4_[4]
        double _x5_[4]
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
        double tmp, sc, err, h_opt, atol, rtol
        double q
        SplineAimAssister f
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
        """Dp54BSpline.__init__(atol,rtol)

        Constructor for the trivariate B-spline interpolated trajectory
        approximation extension type.

        param: atol -- Absolute tolerance for the Dormand-Prince 5(4) method
        param: rtol -- Relative tolerance for the Dormand-Prince 5(4) method

        """
        self.atol = atol
        self.rtol = rtol

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def set_aim_assister(self, SplineAimAssister direction_generator):
        """Dp54BSpline.set_aim_assister(direction_generator)

        Sets the aim assister. Must be called *before* Dp54BSpline.__call__.

        param: direction_generator -- SplineAimAssister instance, initialized
                                      with a SplineEigenvectorInterpolator
                                      object as well as a target

        """
        self.f = direction_generator
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def unset_aim_assister(self):
        """Dp54BSpline.unset_aim_assister()

        Unsets the aim_assister. Useful in order to ensure that the aim assister
        is always updated when it should be, namely in terms of a new target
        or, for that matter, a new vector field.

        """
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos, double h):
        """Dp54BSpline.__call__(t,pos,h)

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
            raise RuntimeError('Dormand-Prince 5(4) B-spline solver not'\
                    ' initialized with a SplineAimAssister instance!')
        dcopy(3,pos,1,pos_i,1)
        self._ev_(&t,pos_i,&h)
        return t, np.copy(self._pos_i_), h

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _ev_(self, double *t, double[::1] pos, double *h):
        """Dp54BSpline._ev_(t,pos,h)

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
            double[::1] k1 = self.k1, k2 = self.k2, k3 = self.k3, k4 = self.k4,\
                        k5 = self.k5, k6 = self.k6, k7 = self.k7, \
                        k_tmp = self.k_tmp
            double[::1] x4 = self.x4, x5 = self.x5
            double[::1] b4 = self.b4, b5 = self.b5
            double[::1] a1 = self.a1, a2 = self.a2, a3 = self.a3, a4 = self.a4,\
                        a5 = self.a5, a6 = self.a6
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
        daxpy(3,a6[5]*h[0],k5,1,k_tmp,1)

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

