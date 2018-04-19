"""This module contains a set of C-level functions intended to accelerate
the computation of geodesic level set approximations to invariant manifolds
of vector fields in R^3.

Methods defined here:

    cy_normalize(v)
    cy_in_plane(pos_curr,prev_point,tan_vec,prev_vec,plane_tol)
    cy_parallel_component(u,v)
    cy_orthogonal_component(u,v)
    cy_cross_product(u,v)
    cy_compute_pos_aim(prev_pos,dist,prev_prev_vec,tan_vec,ang_offset)
    cy_norm2(u)
    cy_min(a,b)

Written by Arne Magnus T. Løken as part of my master's thesis work in physics
at NTNU, spring 2018.

"""

cimport cython

cimport numpy as np
import numpy as np

from scipy.linalg.cython_blas cimport ddot as scp_ddot, dnrm2 as scp_dnrm2, daxpy as scp_daxpy, dcopy as scp_dcopy, dscal as scp_dscal


from libc.math cimport fabs as c_fabs, tan as c_tan

@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef void daxpy(int N, double alpha, double[::1] &x, int INCX, double[::1] &y, int INCY):
    # y <- ax + y
    scp_daxpy(&N, &alpha, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef void dcopy(int N, double[::1] &x, int INCX, double[::1] &y, int INCY):
    # y <- x
    scp_dcopy(&N, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef double ddot(int N, double[::1] &x, int INCX, double[::1] &y, int INCY):
    # return xT*y
    return scp_ddot(&N, &x[0], &INCX, &y[0], &INCY)

@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef double dnrm2(int N, double[::1] &x, int INCX):
    # return (xT*t)**0.5
    return scp_dnrm2(&N, &x[0], &INCX)

@cython.initializedcheck(False)
@cython.boundscheck(False)
cdef void dscal(int N, double alpha, double[::1] &x, int INCX):
    # x <- alpha*x
    scp_dscal(&N, &alpha, &x[0], &INCX)

@cython.cdivision(True)
@cython.initializedcheck(False)
cdef void _cy_normalize_(double[::1] v):
    cdef:
        int N = v.shape[0]
        int INCX = 1
    while(dnrm2(N,v,INCX) < 0.0001):
        dscal(N,100,v,INCX)
    dscal(N,1/dnrm2(N,v,INCX),v,INCX)

@cython.cdivision(False)
@cython.initializedcheck(False)
def cy_normalize(double[::1] v):
    """cy_normalize(v)

    Normalizes a vector in R^3.

    param: Three-component (C-contiguous) NumPy array, containing the vector
           components

    return: Normalized vector, as a three-component (C-contiguous) NumPy array

    """
    cdef:
        double _ret_[3]
        double[::1] ret = _ret_
    dcopy(3,v,1,ret,1)
    _cy_normalize_(ret)
    return np.copy(ret)


@cython.initializedcheck(False)
def cy_in_plane(double[::1] pos_curr, double[::1] prev_point, double[::1] tan_vec, double[::1] prev_vec, double plane_tol):
    """cy_in_plane(pos_curr,prev_point,tan_vec,prev_vec,plane_tol)

    Determines whether or not 'pos_curr' lies within the half-plane defined
    as being orthogonal to 'tan_vec', passing through 'prev_point' and
    extending radially outwards from there, i.e., in the direction specified
    by 'prev_vec', within the tolerance level plane_tol.

    param: pos_curr --   Three-component (C-contiguous) NumPy array, containing
                         the (Cartesian) coordinates of the point which may
                         or may not be within the aforementioned half-plane
    param: prev_point -- Three-component (C-contiguous) NumPy array, containing
                         the (Cartesian) coordinates of the point at which
                         the aforementioned half-plane originates
    param: tan_vec --    Three-component (C-contiguous) NumPy array, containing
                         the components of a normalized (approximately) tangential
                         vector at prev_point
    param: prev_vec --   Three-component (C-contiguous) NumPy array, containing
                         the components of the vector from prev_point to its
                         corresponding point in the previous geodesic level set
    param: plane_tol --  Double-precision float giving the tolerance level for
                         whether or not a point lies within a plane

    return: t_f -- Boolean flag indicating whether or not the point lies within
                   the plane, subject to the prescribed tolerance level.

    """
    cdef:
        double _tmp_[3]
        double[::1] tmp = _tmp_
    dcopy(3,prev_point,1,tmp,1)
    dscal(3,-1,tmp,1)
    daxpy(3,1,pos_curr,1,tmp,1)
    _cy_normalize_(tmp)
    return (c_fabs(ddot(3,tmp,1,tan_vec,1)) < plane_tol) and (ddot(3,tmp,1,prev_vec,1) > 0)

@cython.initializedcheck(False)
@cython.cdivision(True)
def cy_parallel_component(double[::1] u, double[::1] v):
    """cy_parallel_component(u,v)

    Computes the component of the vector u, which is parallel to the vector v.

    param: u -- Three-component (C-contiguous) NumPy array, containing the
                components of u
    param: v -- Three-component (C-contiguous) NumPy array, containing the
                components of v

    return: vec -- Three-component (C-contiguous) NumPy array, containing the
                   parallel component of u along v.

    """
    cdef:
        double _ret_[3]
        double[::1] ret = _ret_
    _cy_parallel_component_(u,v,ret)
    return np.copy(ret)

@cython.initializedcheck(False)
@cython.cdivision(True)
cdef void _cy_parallel_component_(double[::1] u, double[::1] v, double[::1] ret):
    cdef:
        double fac
    fac = ddot(3,u,1,v,1)/ddot(3,v,1,v,1)
    dscal(3,0,ret,1)
    daxpy(3,fac,v,1,ret,1)

@cython.initializedcheck(False)
@cython.cdivision(True)
def cy_dot(double[::1] u, double[::1] v):
    return ddot(3,u,1,v,1)

@cython.initializedcheck(False)
def cy_orthogonal_component(double[::1] u, double[::1] v):
    """cy_orthogonal_component(u,v)

    Computes the component of the vector u, which is orthogonal to the vector v.

    param: u -- Three-component (C-contiguous) NumPy array, containing the
                components of u
    param: v -- Three-component (C-contiguous) NumPy array, containing the
                components of v

    return: vec -- Three-component (C-contiguous) NumPy array, containing the
                   component of u which is orthogonal to v.

    """
    cdef:
        double _ret_[3]
        double[::1] ret = _ret_
    _cy_orthogonal_component_(u,v,ret)
    return np.copy(ret)

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
def cy_cross_product(double[::1] u, double[::1] v):
    """cy_cross_product(u,v)

    Computes the vector product between the vectors u and v in R^3

    param: u -- Three-component (C-contiguous) NumPy array, containing the
                components of u
    param: v -- Three-component (C-contiguous) NumPy array, containing the
                components of v

    return: vec -- Three-component (C-contiguous) NumPy array, containing the
                   components of the cross product between u and v.

    """
    cdef:
        double _ret_[3]
        double[::1] ret = _ret_
    _cy_cross_product_(u,v,ret)
    return np.copy(ret)

@cython.initializedcheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _cy_cross_product_(double[::1] u, double[::1] v, double[::1] ret):
    ret[0] = u[1]*v[2]-u[2]*v[1]
    ret[1] = -(u[0]*v[2]-u[2]*v[0])
    ret[2] = u[0]*v[1]-u[1]*v[0]


@cython.initializedcheck(False)
def cy_compute_pos_aim(double[::1] prev_pos, double dist, double[::1] prev_prev_vec, double[::1] tan_vec, double ang_offset):
    """cy_compute_pos_aim(prev_pos,dist,prev_prev_vec,tan_vec,ang_offset)

    Computes a point to aim at, for the development of new geodesic level sets.

    param: prev_pos --      Three-component (C-contiguous) NumPy array, containing
                            the (Cartesian) coordinates of the point in the previous
                            geodesic level set, from which a point in a new
                            geodesic level set is to be developed
    param: dist --          The (Euclidean) distance from prev_pos, at which a
                            new point is sought
    param: prev_prev_vec -- Three-component (C-contiguous) NumPy array,
                            containing the components of a vector which defines
                            an approximately radially outwards direction from
                            prev_pos
    param: tan_vec --       Three-component (C-contiguous) NumPy array,
                            containing the components of a normalized vector
                            which is approximately tangent to 'prev_pos',
                            and functions as a normal vector to the half-plane
                            in which a new point is sought
    param: ang_offset --    Double-precision float, giving the angular offset
                            (in radians) between prev_prev_vec and the vector
                            from prev_pos to the aiming point

    return: pos_aim -- Three-component (C-contiguous) NumPy array, containing
                       the (Cartesian) coordinates at the point to aim at.

    """
    cdef:
        double _ret_[3]
        double _up_[3]
        double _out_[3]
        double _tmp_[3]
        double[::1] ret = _ret_, up = _up_, out = _out_, tmp = _tmp_
    _cy_orthogonal_component_(prev_prev_vec, tan_vec, out)
    _cy_normalize_(out)
    _cy_cross_product_(tan_vec,out,up)
    _cy_normalize_(up)
    dcopy(3,up,1,tmp,1)
    dscal(3,c_tan(ang_offset),tmp,1)
    daxpy(3,1,out,1,tmp,1)
    _cy_normalize_(tmp)
    dcopy(3,prev_pos,1,ret,1)
    daxpy(3,dist,tmp,1,ret,1)
    return np.copy(ret)


@cython.initializedcheck(False)
def cy_norm2(double[::1] u):
    """cy_norm2(u)

    Computes the 2-norm of an n-vector.

    param: u -- n-component (C-contiguous) NumPy array, containing the
                components of u

    return: nrm -- 2-norm of u, as a double-precision float.

    """
    cdef:
        int N = u.shape[0]
        int INCX = 1
    return dnrm2(N,u,1)


def cy_min(double a, double b):
    """cy_min(a,b)

    Returns the smallest of two double-precision numbers.

    param: a -- Double-precision number
    param: b -- Double-precision number

    return: r -- Double-precision float, the smallest number of a and b.

    """
    if b < a:
        return b
    return a

def cy_max(double a, double b):
    """cy_max(a,b)

    Returns the largest of two double-precision numbers.

    param: a -- Double-precision number
    param: b -- Double-precision number

    return: r -- Double-precision float, the smallest number of a and b.

    """
    if b > a:
        return b
    return a
