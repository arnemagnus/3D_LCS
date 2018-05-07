cimport numpy as np
import numpy as np

cimport cython

from scipy.linalg.cython_blas cimport ddot as scp_ddot, dnrm2 as scp_dnrm2,\
                                      daxpy as scp_daxpy, dscal as scp_dscal,\
                                      dcopy as scp_dcopy

from libc.math cimport pow as c_pow, fmod as c_fmod, floor as c_floor, \
                       copysign as c_copysign, fabs as c_fabs, \
                       sin as c_sin, cos as c_cos

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




@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void _cy_parallel_component_(double[::1] u, double[::1] v, double[::1] ret):
    cdef:
        double fac
    fac = ddot(3,u,1,v,1)/ddot(3,v,1,v,1)
    dscal(3,0,ret,1)
    daxpy(3,fac,v,1,ret,1)



@cython.initializedcheck(False)
cdef void _cy_orthogonal_component_(double[::1] u, double[::1] v, double[::1] ret):
    cdef:
        double _tmp_[3]
        double[::1] tmp = _tmp_
    dcopy(3,u,1,ret,1)
    _cy_parallel_component_(u,v,tmp)
    daxpy(3,-1,tmp,1,ret,1)



@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _cy_cross_product_(double[::1] u, double[::1] v, double[::1] &ret):
    ret[0] = u[1]*v[2]-u[2]*v[1]
    ret[1] = -(u[0]*v[2]-u[2]*v[0])
    ret[2] = u[0]*v[1]-u[1]*v[0]

@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _cy_normalize_(double[::1] v):
    cdef:
        int N = v.shape[0]
        int INCX = 1
    while(dnrm2(N,v,INCX) < 0.0001):
        dscal(N,100,v,INCX)
    dscal(N,1/dnrm2(N,v,INCX),v,INCX)

cdef class CallableFunction:
    def __cinit__(self):
        pass
    def __init__(self):
        pass
    cdef _ev_(self, double[::1] x, double[::1] &ret):
        pass

cdef class SinusoidalField(CallableFunction):
    cdef:
        double freq_x, freq_y, ampl
        double _ret_[3]
        double[::1] ret_mv

    def __cinit__(self):
        self.ret_mv = self._ret_

    def __init__(self, double freq_x = 1., double freq_y = 1.,
            double ampl = 1.):
        self.freq_x = freq_x
        self.freq_y = freq_y
        self.ampl = ampl

    @cython.initializedcheck(False)
    def __call__(self, double[::1] x not None):
        cdef:
            double[::1] ret = self.ret_mv
        self._ev_(x,ret)
        return np.copy(ret)


    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _ev_(self, double[::1] x, double[::1] &ret):
        ret[0] = self.ampl*self.freq_x*c_cos(self.freq_x*x[0])*c_sin(self.freq_y*x[1])
        ret[1] = self.ampl*self.freq_y*c_sin(self.freq_x*x[0])*c_cos(self.freq_y*x[1])
        ret[2] = -1
        _cy_normalize_(ret)


cdef class SphericalField(CallableFunction):
    cdef:
        double _ret_[3]
        double[::1] ret_mv
        double _orig_[3]
        double[::1] origin
    def __cinit__(self):
        self.ret_mv = self._ret_
        self.origin = self._orig_
    @cython.initializedcheck(False)
    def __init__(self, double[::1] origin not None):
        if origin.shape[0] != 3:
            raise RuntimeError('Don''t fuck with squirrels, Morty!')
        dcopy(3,origin,1,self.origin,1)
    @cython.initializedcheck(False)
    def __call__(self, double[::1] x not None):
        cdef:
            double[::1] ret = self.ret_mv
        self._ev_(x,ret)
        return np.copy(ret)
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef _ev_(self, double[::1] x, double[::1] &ret):
        dcopy(3,x,1,ret,1)
        daxpy(3,-1,self.origin,1,ret,1)
        _cy_normalize_(ret)

cdef class AnalyticalAimAssister:
    """A Cython extension type tailor-made for computing trajectories defined
    to be pointwise orthogonal to two vectors in 3D.

    Methods defined here:

    SplineAimAssister.__init__(CallableFunction f)
    SplineAimAssister.set_tangential_vec(target)
    SplineAimAssister.unset_target()
    SplineAimAssister.__ev__(t,pos)

    Version: 0.2

    """
    cdef:
        double _tan_vec_[3]
        double[::1] tan_vec
        double _prev_vec_[3]
        double[::1] prev_vec
        double _xi_[3]
        double[::1] xi
        double _pos_[3]
        double[::1] pos
        double[::1] ret_mv
        CallableFunction xi_fun
        bint initialized_tan_vec
        bint initialized_prev_vec

    def __cinit__(self):
        self.tan_vec = self._tan_vec_
        self.prev_vec = self._prev_vec_
        self.xi = self._xi_
        self.pos = self._pos_
        self.ret_mv = np.empty(3)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __init__(self, CallableFunction xi):
        """SplineAimAssister(xi)

        Constructor for the trivariate B-spline aim assister extension type.

        param: xi     -- A CallableFunction instance of the (normalized)
                         vector field which locally defines the normal vector
                         to the planes in which one is permitted to aim.

        """
        self.xi_fun = xi

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef void set_tan_vec(self, double[::1] tan_vec):
        if tan_vec.shape[0] != 3:
            raise RuntimeError('The interpolation-aiming routine is custom-built'\
                    +' for three dimensional data!')
        dcopy(3,tan_vec,1,self.tan_vec,1)
        self.initialized_tan_vec = True

    cpdef void unset_tan_vec(self):
        """SplineAimAssister.unset_target()

        Unsets the target. Useful in order to ensure that the target is always
        updated when it should be.

        """
        self.initialized_tan_vec = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef void set_prev_vec(self, double[::1] prev_vec):
        if prev_vec.shape[0] != 3:
            raise RuntimeError('The interpolation-aiming routine is custom-built'\
                    +' for three dimensional data!')
        dcopy(3,prev_vec,1,self.prev_vec,1)
        _cy_normalize_(self.prev_vec)
        self.initialized_prev_vec = True

    cpdef void unset_prev_vec(self):
        """SplineAimAssister.unset_target()

        Unsets the target. Useful in order to ensure that the target is always
        updated when it should be.

        """
        self.initialized_prev_vec = False


    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos not None):
        cdef:
            double[::1] ret = self.ret_mv
        self._ev_(t,pos,ret)
        return np.copy(ret)

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef _ev_(self, double t, double[::1] &pos_, double[::1] &ret):
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
            double[::1] xi = self.xi, pos = self.pos
        if pos_.shape[0] != 3:
            raise RuntimeError('The aiming routine is custom-built'\
                    +' for three dimensional data!')

        if not (self.initialized_tan_vec and self.initialized_prev_vec):
            raise RuntimeError('Aim assister not initialized!')
        dcopy(3,pos_,1,pos,1)

        self.xi_fun._ev_(pos, xi)
        if 1 - c_pow(ddot(3,xi,1,self.tan_vec,1),2) < 1e-10:
            dcopy(3,self.prev_vec,1,ret,1)
        else:
            _cy_cross_product_(xi,self.tan_vec,ret)
        _cy_normalize_(ret)
        if ddot(3,ret,1,self.prev_vec,1) < 0:
            dscal(3,-1,ret,1)


    def __dealoc__(self):
        pass

cdef class Dp87Analytical:
    """A Cython extension type tailor-made for computing trajectories which
    are defined as being orthogonal to a linearly interpolated trivariate vector
    field in R^3 by means of the Dormand-Prince 8(7) numerical integrator.

    Methods defined here:

    Dp87Analytical.__init__(atol, rtol)
    Dp87Analytical.set_aim_assister(direction_generator)
    Dp87Analytical.unset_aim_assister()

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
        AnalyticalAimAssister f
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
        """Dp87Analytical.__init__(atol,rtol)

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
    def set_aim_assister(self, AnalyticalAimAssister direction_generator):
        """Dp87Analytical.set_aim_assister(direction_generator)

        Sets the aim assister. Must be called *before* Dp87Analytical.__call__.

        param: direction_generator -- AnalyticalAimAssister instance,
                                      initialized with a CallableFunction
                                      object as well as a target

        """
        self.f = direction_generator
        self.initialized = True

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def unset_aim_assister(self):
        """Dp87Analytical.unset_aim_assister()

        Unsets the aim_assister. Useful in order to ensure that the aim assister
        is always updated when it should be, namely in terms of a new target
        or, for that matter, a new vector field.

        """
        self.initialized = False

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def __call__(self, double t, double[::1] pos, double h):
        """Dp87Analytical.__call__(t,pos,h)

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
                       (a) Dormand-Prince 8(7) Analytical approximation of the
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
                    ' initialized with a AnalyticalAimAssister instance!')
        dcopy(3,pos,1,pos_i,1)
        self._ev_(&t,pos_i,&h)
        return t, np.copy(self._pos_i_), h

    @cython.initializedcheck(False)
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    cdef void _ev_(self, double *t, double[::1] pos, double *h):
        """Dp87Analytical._ev_(t,pos,h)

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

        self.f._ev_(t[0],pos,k1)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a1[0]*h[0],k1,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k2)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a2[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a2[1]*h[0],k2,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k3)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a3[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a3[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a3[2]*h[0],k3,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k4)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a4[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a4[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a4[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a4[3]*h[0],k4,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k5)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a5[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a5[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a5[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a5[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a5[4]*h[0],k5,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k6)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a6[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a6[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a6[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a6[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a6[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a6[5]*h[0],k6,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k7)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a7[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a7[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a7[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a7[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a7[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a7[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a7[6]*h[0],k7,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k8)
        dcopy(3,pos,1,k_tmp,1)
        daxpy(3,a8[0]*h[0],k1,1,k_tmp,1)
        daxpy(3,a8[1]*h[0],k2,1,k_tmp,1)
        daxpy(3,a8[2]*h[0],k3,1,k_tmp,1)
        daxpy(3,a8[3]*h[0],k4,1,k_tmp,1)
        daxpy(3,a8[4]*h[0],k5,1,k_tmp,1)
        daxpy(3,a8[5]*h[0],k6,1,k_tmp,1)
        daxpy(3,a8[6]*h[0],k7,1,k_tmp,1)
        daxpy(3,a8[7]*h[0],k8,1,k_tmp,1)

        self.f._ev_(t[0],k_tmp,k9)
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

        self.f._ev_(t[0],k_tmp,k10)
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

        self.f._ev_(t[0],k_tmp,k11)
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

        self.f._ev_(t[0],k_tmp,k12)
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

        self.f._ev_(t[0],k_tmp,k13)

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
