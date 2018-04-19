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

cdef class Triangle3D:
    cdef:
        double _vertex0_[3]
        double _vertex1_[3]
        double _vertex2_[3]
        double[::1] vertex0, vertex1, vertex2
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __cinit__(self):
        self.vertex0 = self._vertex0_
        self.vertex1 = self._vertex1_
        self.vertex2 = self._vertex2_
    @cython.initializedcheck(False)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, double[:,::1] vertices not None):
        """Constructor for a Triangle3D object.

        param: vertices -- A (C-contiguous) NumPy array of shape (3,3),
                           containing the triangle vertices.
                           vertices[i] should contain the [x,y,z] coordinates,
                           in that order.
        """
        if vertices.shape[0] != 3 or vertices.shape[1] != 3:
            raise ValueError('Invalid input. Check constructor docstring.')
        dcopy(3,vertices[0],1,self.vertex0,1)
        dcopy(3,vertices[1],1,self.vertex1,1)
        dcopy(3,vertices[2],1,self.vertex2,1)

cdef class MollerTrumboreChecker:
    cdef:
        double isect_eps, prl_eps
        bint culling
        double _edge1_[3]
        double _edge2_[3]
        double _pvec_[3]
        double _tvec_[3]
        double _qvec_[3]
        double _ray_origin_[3]
        double _ray_direction_[3]
        double _isect_vec_[3]
        double _diffvec1_[3]
        double _diffvec2_[3]
        double[::1] edge1, edge2, pvec, tvec, qvec, ray_origin, ray_direction, \
                    isect_vec, diffvec1, diffvec2
        double u, v, det, inv_det, t
    def __cinit__(self):
        self.edge1 = self._edge1_
        self.edge2 = self._edge2_
        self.pvec = self._pvec_
        self.tvec = self._tvec_
        self.qvec = self._qvec_
        self.ray_origin = self._ray_origin_
        self.ray_direction = self._ray_direction_
        self.isect_vec = self._isect_vec_
        self.diffvec1 = self._diffvec1_
        self.diffvec2 = self._diffvec2_
        self.prl_eps = 1e-8
    def __init__(self, double eps = 1e-8, bint culling = False):
        """Constructor for a MollerTrumboreChecker object.

        OPTIONAL:
        param: eps -- Epsilon. Default: 1e-8.
        param: culling -- Boolean flag indicating whether or not culling is
                          desired. Default: False
        """
        if eps < 0:
            raise ValueError('Eps cannot be negative!')
        self.isect_eps = eps
        self.culling = culling
    def __call__(self, Triangle3D a not None, Triangle3D b not None):
        return self.intersects(a,b) or self.intersects(b,a)

    @cython.initializedcheck(False)
    cdef bint intersects(self, Triangle3D a, Triangle3D b):
        cdef:
            double[::1] ray_origin = self.ray_origin, \
                        ray_direction = self.ray_direction
        # Check one of three
        dcopy(3,b.vertex0,1,ray_origin,1)
        dcopy(3,b.vertex1,1,ray_direction,1)
        daxpy(3,-1,b.vertex0,1,ray_direction,1) # ray_dir = b.vertex1 - b.vertex0
        if self._ray_triangle_isect_(a, ray_origin, ray_direction):
            return True
        # Check two of three
        dcopy(3,b.vertex2,1,ray_direction,1)
        daxpy(3,-1,b.vertex0,1,ray_direction,1) # ray_dir = b.vertex2 - b.vertex0
        if self._ray_triangle_isect_(a, ray_origin, ray_direction):
            return True
        # Check three of three
        dcopy(3,b.vertex1,1,ray_origin,1)
        dcopy(3,b.vertex2,1,ray_direction,1)
        daxpy(3,-1,b.vertex1,1,ray_direction,1) # ray_dir = b.vertex2 - b.vertex1
        if self._ray_triangle_isect_(a, ray_origin, ray_direction):
            return True
        # None of the rays in triangle b intersect triangle a, hence
        return False

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef bint _ray_triangle_isect_(self, Triangle3D a, \
                                         double[::1] ray_origin, \
                                         double[::1] ray_direction\
                                  ):
        cdef:
            double[::1] edge1 = self.edge1, edge2 = self.edge2,\
                        pvec = self.pvec, tvec = self.tvec, \
                        qvec = self.qvec, isect_vec = self.isect_vec, \
                        diffvec1 = self.diffvec1, diffvec2 = self.diffvec2
            double isect_eps = self.isect_eps, prl_eps = self.prl_eps, \
                   det = self.det, u = self.u, v = self.v, \
                   t = self.t, inv_det = self.inv_det
        dcopy(3,a.vertex1,1,edge1,1)
        daxpy(3,-1,a.vertex0,1,edge1,1) # edge1 = a.vertex1 - a.vertex0
        dcopy(3,a.vertex2,1,edge2,1)
        daxpy(3,-1,a.vertex0,1,edge2,1) # edge2 = a.vertex2 - a.vertex0
        _cy_cross_product_(ray_direction, edge2, pvec) # p = ray_dir x edge_2
        det = ddot(3,edge1,1,pvec,1)
        if self.culling:
            if det < isect_eps:
                return False
            dcopy(3,ray_origin,1,tvec,1)
            daxpy(3,-1,a.vertex0,1,tvec,1) # tvec = ray_origin - a.vertex0
            u = ddot(3,tvec,1,pvec,1)
            if (u < 0 or u > det):
                return False
            _cy_cross_product_(tvec,edge1,qvec) # q = tvec x edge1
            v = ddot(3,ray_direction,1,qvec,1)
            if (v < 0 or u + v > det):
                return False
            t = ddot(3,edge2,1,qvec,1)
            inv_det = 1/det
            t *= inv_det
            if t < isect_eps:
                return False
        else:
            if (det < isect_eps and det > -isect_eps):
                return False
            dcopy(3,ray_origin,1,tvec,1)
            daxpy(3,-1,a.vertex0,1,tvec,1) # tvec = ray_origin - a.vertex0
            inv_det = 1/det
            u = ddot(3,tvec,1,pvec,1)*inv_det
            if (u < 0 or u > 1):
                return False
            _cy_cross_product_(tvec,edge1,qvec) # q = tvec x edge1
            v = ddot(3,ray_direction,1,qvec,1)*inv_det
            if (v < 0 or u + v > 1):
                return False
            t = ddot(3,edge2,1,qvec,1)*inv_det
            if t < isect_eps:
                return False
        dcopy(3,ray_origin,1,isect_vec,1)
        daxpy(3,-1,a.vertex0,1,isect_vec,1) # isect_vec = ray_origin - a.vertex0
        daxpy(3,t,ray_direction,1,isect_vec,1)
        if dnrm2(3,isect_vec,1) < prl_eps:
            return False
        else:
            _cy_normalize_(isect_vec)
            _cy_normalize_(edge1)
            _cy_normalize_(edge2)
            dcopy(3,isect_vec,1,diffvec1,1)
            daxpy(3,-1,edge1,1,diffvec1,1) # diffvec 1 = isect_vec - edge1
            dcopy(3,isect_vec,1,diffvec2,1)
            daxpy(3,-1,edge2,1,diffvec2,1) # diffvec 2 = isect_vec - edge2
            if dnrm2(3,diffvec1,1) < prl_eps or dnrm2(3,diffvec2,1) < prl_eps:
                return False
        return True

