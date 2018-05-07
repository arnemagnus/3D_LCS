"""
This module contains a set of C-level functions and extension types intended
to accelerate the check for self-intersections when developing geodesic level
sets.

Extension types defined here:
    Triangle3D(vertices)
    MollerTrumboreChecker(eps,culling)

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
                       copysign as c_copysign, fabs as c_fabs, \
                       sin as c_sin, cos as c_cos


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
    # y <- ax + y
    scp_daxpy(&N, &alpha, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void dscal(int N, double alpha, double[::1] &x, int INCX) nogil:
    # x <- ax
    scp_dscal(&N, &alpha, &x[0], &INCX)



@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _cy_cross_product_(double[::1] u, double[::1] v, double[::1] &ret):
    # ret <- u x v
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

cdef class Triangle3D:
    """
    A Cython extension type which remembers the vertices of a three-dimensional
    triangle.

    Methods defined here:
        Triangle3D.__init__(vertices)

    Version: 0.1

    """
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
        """
        Triangle3D.__init__(vertices)

        Constructor for a Triangle3D object.

        Parameters
        ----------
        vertices : (3,3) array_like
            A (C-contiguous) NumPy array of shape (3,3) and type np.float64,
            containing the triangle vertices. All vertices[i] should contain
            the [x,y,z] coordinates, in that order.

        """
        if vertices.shape[0] != 3 or vertices.shape[1] != 3:
            raise ValueError('Invalid input. Check constructor docstring.')
        dcopy(3,vertices[0],1,self.vertex0,1)
        dcopy(3,vertices[1],1,self.vertex1,1)
        dcopy(3,vertices[2],1,self.vertex2,1)


cdef class MollerTrumboreChecker:
    """
    A Cython extension type which facilitates pairwise comparison between
    Triangle3D instances, in order to detect intersections between them.

    In its current state, the underlying algorithm does not flag for
    intersections when
        1) The triangles share at least one corner
        2) The triangles share at least one edge
        3) Both intersections occur along the edges
        4) The triangles are identical

    The above is based on the intended use; namely, checking whether or not
    self-intersections in geodesic level sets are 'severe' enough so that
    there is no good reason to continue developing manifolds.

    Methods defined here:
        MollerTrumboreChecker.__init__(eps,culling)
        MollerTrumboreChecker.__call__(tri1,tri2)

    Version: 0.1

    """
    cdef:
        double isect_eps, prl_eps
        bint culling
        double _edge1_[3]
        double _edge2_[3]
        double _edge3_[3]
        double _pvec_[3]
        double _tvec_[3]
        double _qvec_[3]
        double _ray_origin_[3]
        double _ray_direction_[3]
        double _isect_pt_[3]
        double _isect_vec_1_[3]
        double _isect_vec_2_[3]
        double _diffvec1_[3]
        double _diffvec2_[3]
        double _diffvec3_[3]
        double _normvec_[3]
        double[::1] edge1, edge2, edge3, pvec, tvec, qvec, \
                    ray_origin, ray_direction, isect_pt, \
                    isect_vec_1, isect_vec_2, diffvec1, diffvec2, diffvec3
        double u, v, det, inv_det, t
    def __cinit__(self):
        self.edge1 = self._edge1_
        self.edge2 = self._edge2_
        self.edge3 = self._edge3_
        self.pvec = self._pvec_
        self.tvec = self._tvec_
        self.qvec = self._qvec_
        self.ray_origin = self._ray_origin_
        self.ray_direction = self._ray_direction_
        self.isect_pt = self._isect_pt_
        self.isect_vec_1 = self._isect_vec_1_
        self.isect_vec_2 = self._isect_vec_2_
        self.diffvec1 = self._diffvec1_
        self.diffvec2 = self._diffvec2_
        self.diffvec3 = self._diffvec3_
        self.prl_eps = 1e-8
    def __init__(self, double eps = 1e-8, bint culling = False):
        """
        MollerTrumboreChecker.__init__(eps,culling)

        Constructor for a MollerTrumboreChecker object.

        Parameters
        ----------
        eps : double, optional
            Epsilon for ray-triangle intersection detection.
            See Möller & Trumbore (1997).
            Default: eps = 1e-8.
        culling : boolean, optional
            Boolean flag indicating whether or not culling is
            desired. Default: False

        """
        if eps < 0:
            raise ValueError('Eps cannot be negative!')
        self.isect_eps = eps
        self.culling = culling
    def __call__(self, Triangle3D tri1 not None, Triangle3D tri2 not None):
        """
        MollerTrumboreChecker.__call__(tri1,tri2)

        Checks whether or not the Triangle3D objects tri1 and tri2 intersect
        one another, subject to the exceptions mentioned in the constructor
        docstring.

        Parameters
        ----------
        tri1 : Triangle3D
            One of the two triangles to compare
        tri2 : Triangle3D
            The second of the two triangles to compare

        Returns
        -------
        tf : boolean
            Boolean flag indicating whether or not the triangles intersect

        """
        return self.intersects(tri1,tri2) or self.intersects(tri2,tri1)
    @cython.initializedcheck(False)
    cdef bint intersects(self, Triangle3D a, Triangle3D b):
        cdef:
            double[::1] ray_origin = self.ray_origin, \
                        ray_direction = self.ray_direction, \
                        isect_pt = self.isect_pt, \
                        sepvec_1 = self.diffvec2, sepvec_2 = self.diffvec3

        # If the triangles share a vertex, don't bother
        dcopy(3,b.vertex0,1,sepvec_1,1)
        daxpy(3,-1,a.vertex0,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex0,1,sepvec_1,1)
        daxpy(3,-1,a.vertex1,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex0,1,sepvec_1,1)
        daxpy(3,-1,a.vertex2,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex1,1,sepvec_1,1)
        daxpy(3,-1,a.vertex0,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex1,1,sepvec_1,1)
        daxpy(3,-1,a.vertex1,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex1,1,sepvec_1,1)
        daxpy(3,-1,a.vertex2,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex2,1,sepvec_1,1)
        daxpy(3,-1,a.vertex0,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex2,1,sepvec_1,1)
        daxpy(3,-1,a.vertex1,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False
        dcopy(3,b.vertex2,1,sepvec_1,1)
        daxpy(3,-1,a.vertex2,1,sepvec_1,1)
        if dnrm2(3,sepvec_1,1) < 1e-8:
            return False

        # Check one of three (ray 1 of triangle b compared to triangle a)
        dcopy(3,b.vertex0,1,ray_origin,1)
        dcopy(3,b.vertex1,1,ray_direction,1)
        daxpy(3,-1,b.vertex0,1,ray_direction,1) # ray_dir = b.vertex1 - b.vertex0
        if self._ray_triangle_isect_(a, ray_origin, ray_direction):
            # Ray intersects triangle body
            # Need to check if the vertices of triangle b from which
            # the ray direction was computed lie on opposite sides of the
            # surface of a.
            # Check if vectors from aforementioned vertices of triangle b
            # have opposing directions
            dcopy(3,isect_pt,1,sepvec_1,1)
            daxpy(3,-1,b.vertex0,1,sepvec_1,1)
            dcopy(3,isect_pt,1,sepvec_2,1)
            daxpy(3,-1,b.vertex1,1,sepvec_2,1)
            if ddot(3,sepvec_1,1,sepvec_2,1) < 0:
                return True

        # Check two of three (ray 2 of triangle b compared to triangle a)
        dcopy(3,b.vertex2,1,ray_direction,1)
        daxpy(3,-1,b.vertex0,1,ray_direction,1) # ray_dir = b.vertex2 - b.vertex0
        if self._ray_triangle_isect_(a, ray_origin, ray_direction):
            # Ray intersects triangle body
            # Need to check if the vertices of triangle b from which
            # the ray direction was computed lie on opposite sides of the
            # surface of a.
            # Check if vectors from aforementioned vertices of triangle b
            # have opposing directions
            dcopy(3,isect_pt,1,sepvec_1,1)
            daxpy(3,-1,b.vertex0,1,sepvec_1,1)
            dcopy(3,isect_pt,1,sepvec_2,1)
            daxpy(3,-1,b.vertex2,1,sepvec_2,1)
            if ddot(3,sepvec_1,1,sepvec_2,1) < 0:
                return True
        # Check three of three (ray 3 of triangle b compared to triangle a)
        dcopy(3,b.vertex1,1,ray_origin,1)
        dcopy(3,b.vertex2,1,ray_direction,1)
        daxpy(3,-1,b.vertex1,1,ray_direction,1) # ray_dir = b.vertex2 - b.vertex1
        if self._ray_triangle_isect_(a, ray_origin, ray_direction):
            # Ray intersects triangle body
            # Need to check if the vertices of triangle b from which
            # the ray direction was computed lie on opposite sides of the
            # surface of a.
            # Check if vectors from aforementioned vertices of triangle b
            # have opposing directions
            dcopy(3,isect_pt,1,sepvec_1,1)
            daxpy(3,-1,b.vertex1,1,sepvec_1,1)
            dcopy(3,isect_pt,1,sepvec_2,1)
            daxpy(3,-1,b.vertex2,1,sepvec_2,1)
            if ddot(3,sepvec_1,1,sepvec_2,1) < 0:
                return True
        # b does not pass through the body of a, hence
        return False

    # Cython implementation of the _actual_ Möller-Trumbore algorithm,
    # modified for the caveats mentioned in the constructor docstring
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef bint _ray_triangle_isect_(self, Triangle3D a, \
                                         double[::1] ray_origin, \
                                         double[::1] ray_direction\
                                  ):
        cdef:
            double[::1] edge1 = self.edge1, edge2 = self.edge2, \
                        edge3 = self.edge3, pvec = self.pvec, \
                        tvec = self.tvec, qvec = self.qvec, \
                        isect_pt = self.isect_pt, \
                        isect_vec_1 = self.isect_vec_1, \
                        isect_vec_2 = self.isect_vec_2, \
                        diffvec1 = self.diffvec1, diffvec2 = self.diffvec2, \
                        diffvec3 = self.diffvec3
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
        # Compute intersection point of ray and triangle
        dcopy(3,ray_origin,1,isect_pt,1)
        daxpy(3,t,ray_direction,1,isect_pt,1)
        # Compute vectors from vertices 0 and 1 to the intersection point
        # (in order to check if the intersection point lies along any
        # of the triangle's edges, in which case we _don't_ flag it as an
        # intersection)
        dcopy(3,isect_pt,1,isect_vec_1,1)
        daxpy(3,-1,a.vertex0,1,isect_vec_1,1) # isec_vec_1 = isect_pt - a.vertex0
        dcopy(3,isect_pt,1,isect_vec_2,1)
        daxpy(3,-1,a.vertex1,1,isect_vec_2,1) # isec_vec_2 = isect_pt - a.vertex1
        if dnrm2(3,isect_vec_1,1) < prl_eps or dnrm2(3,isect_vec_2,1) < prl_eps:
            # Intersection _at_ one of the aforementioned vertices ->
            # We don't flag it as such
            return False
        else:
            # Compute the third (and final) triangle edge in order to
            # be able to check whether the suggested intersection lies here
            dcopy(3,a.vertex2,1,edge3,1)
            daxpy(3,-1,a.vertex1,1,edge3,1) # edge3 = a.vertex2 - a.vertex1
            # Normalize vectors from vertices 0 and 1 to the intersection point,
            # as well as the edge vectors (to facilitate comparisons, as they
            # have been computed to be parallel rather than antiparallel)
            _cy_normalize_(isect_vec_1)
            _cy_normalize_(isect_vec_2)
            _cy_normalize_(edge1)
            _cy_normalize_(edge2)
            _cy_normalize_(edge3)
            # Compute vector differences between aforementioned vectors
            dcopy(3,isect_vec_1,1,diffvec1,1)
            daxpy(3,-1,edge1,1,diffvec1,1) # diffvec1 = isect_vec_1 - edge1
            dcopy(3,isect_vec_1,1,diffvec2,1)
            daxpy(3,-1,edge2,1,diffvec2,1) # diffvec2 = isect_vec_1 - edge2
            dcopy(3,isect_vec_2,1,diffvec3,1)
            daxpy(3,-1,edge3,1,diffvec3,1)  # diffvec3 = isect_vec_2 - edge3
            if dnrm2(3,diffvec1,1) < prl_eps or dnrm2(3,diffvec2,1) < prl_eps \
                    or dnrm2(3,diffvec3,1) < prl_eps:
                # Intersection point lies along triangle edge
                return False
        # Ray intersects the triangle within its body
        return True

