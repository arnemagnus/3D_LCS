cimport cython
import time
from cython.parallel cimport parallel, prange

cimport numpy as np
import numpy as np

from libcpp.vector cimport vector

from libc.math cimport ceil as cy_ceil, pow as cy_pow, fabs as cy_fabs,\
                        copysign as cy_copysign

from scipy.linalg.cython_blas cimport daxpy as cy_daxpy

ctypedef vector[double] dbl_vec


ctypedef void (*rhs_func)(double, dbl_vec, dbl_vec&)

cdef void _daxpy_(double alpha, dbl_vec &x, dbl_vec &y) nogil:
    " y <- ax + y"
    cdef:
        int N = x.size()
        int INCX = 1, INCY = 1
    cy_daxpy(&N,
             &alpha,
             &x[0],
             &INCX,
             &y[0],
             &INCY)


cdef void rk3_p(double *t, dbl_vec &x, double *h, rhs_func f):
    cdef:
        int ndim = x.size()
        dbl_vec k1, k2, k3, tmp
        int i

    k1  = dbl_vec(ndim)
    k2  = dbl_vec(ndim)
    k3  = dbl_vec(ndim)


    f(t[0]           , x  , k1)

    tmp = dbl_vec(ndim)
    _daxpy_(0.5*h[0],k1,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0] + 0.5*h[0], tmp, k2)

    tmp = dbl_vec(ndim)
    _daxpy_(-h[0],k1,tmp)
    _daxpy_(2*h[0],k2,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0] + h[0]    , tmp, k3)

    # Using _daxpy_ recursively consistently takes *marginally* longer time
    # than using a for loop for (up to) three unknowns (not tested for more
    # unknowns)
#    tmp = dbl_vec(ndim)
#    _daxpy_(1.,k1,tmp)
#    _daxpy_(4.,k2,tmp)
#    _daxpy_(1.,k3,tmp)
#    _daxpy_(0.16666666666666667*h[0],tmp,x)

    for i in range(ndim):
        x[i] = x[i] + 0.16666666666666667*(k1[i]+4*k2[i]+k3[i])*h[0]

    t[0] = t[0] + h[0]


cdef void rk4_p(double *t, dbl_vec &x, double *h, rhs_func f):
    cdef:
        int ndim = x.size()
        dbl_vec k1, k2, k3, k4, tmp
        int i

    k1  = dbl_vec(ndim)
    k2  = dbl_vec(ndim)
    k3  = dbl_vec(ndim)
    k4  = dbl_vec(ndim)

    f(t[0]           , x  , k1)

    tmp = dbl_vec(ndim)
    _daxpy_(0.5*h[0],k1,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0] + 0.5*h[0], tmp, k2)

    tmp = dbl_vec(ndim)
    _daxpy_(0.5*h[0],k2,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0] + 0.5*h[0], tmp, k3)

    tmp = dbl_vec(ndim)
    _daxpy_(h[0],k3,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0] + h[0]    , tmp, k4)



    # Using _daxpy_ recursively consistently takes *marginally* longer time
    # than using a for loop for (up to) three unknowns (not tested for more
    # unknowns)
#    tmp = dbl_vec(ndim)
#    _daxpy_(1.,k1,tmp)
#    _daxpy_(2.,k2,tmp)
#    _daxpy_(2.,k3,tmp)
#    _daxpy_(1.,k4,tmp)

#    _daxpy_(0.16666666666666667*h[0],tmp,x)

    for i in range(ndim):
        x[i] = x[i] + 0.16666666666666667*(k1[i]+2*k2[i]+2*k3[i]+k4[i])*h[0]

    t[0] = t[0] + h[0]

cdef void rkdp54_p(double *t, dbl_vec &x, double *h, rhs_func f):
    cdef:
        int ndim = x.size()
        dbl_vec k1, k2, k3, k4, k5, k6, k7
        double c[6]
        double a1[1]
        double a2[2]
        double a3[3]
        double a4[4]
        double a5[5]
        double a6[6]
        double b4[7]
        double b5[7]
        double h_opt, err, sc, q, err_tmp
        dbl_vec tmp
        dbl_vec x4, x5
        double atol, rtol
        int i
        double fac, maxfac

    fac = 0.8
    maxfac = 2.

    atol = 1e-5
    rtol = 1e-7

    c  = [       1./5., \
                              3./10., \
                                              4./5., \
                                                          8./9., \
                                                                              1., \
                                                                                          1.]
    a1 = [       1./5.                                                                      ]
    a2 = [      3./40.,        9./40.                                                       ]
    a3 = [     44./45.,      -56./15.,       32./9.                                         ]
    a4 = [19372./6561., -25360./2187., 64448./6561., -212./729.                             ]
    a5 = [ 9017./3168.,     -355./33., 46732./5247.,   49./176.,  -5103./18656.             ]
    a6 = [    35./384.,            0.,   500./1113.,  125./192.,    -2187./6784.,    11./84.]

    b4 = [5179./57600.,            0., 7571./16695.,  393./640., -92097./339200., 187./2100., 1./40.]
    b5 = [    35./384.,            0.,   500./1113.,  125./192.,    -2187./6784.,    11./84.,     0.]

    k1 = dbl_vec(ndim)
    k2 = dbl_vec(ndim)
    k3 = dbl_vec(ndim)
    k4 = dbl_vec(ndim)
    k5 = dbl_vec(ndim)
    k6 = dbl_vec(ndim)
    k7 = dbl_vec(ndim)

    f(t[0], x, k1)

    tmp = dbl_vec(ndim)
    _daxpy_(a1[0]*h[0],k1,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0] + c[0]*h[0], tmp, k2)

    tmp = dbl_vec(ndim)
    _daxpy_(a2[0]*h[0],k1,tmp)
    _daxpy_(a2[1]*h[0],k2,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0] + c[1]*h[0], tmp, k3)

    tmp = dbl_vec(ndim)
    _daxpy_(a3[0]*h[0],k1,tmp)
    _daxpy_(a3[1]*h[0],k2,tmp)
    _daxpy_(a3[2]*h[0],k3,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0]+c[2]*h[0], tmp, k4)

    tmp = dbl_vec(ndim)
    _daxpy_(a4[0]*h[0],k1,tmp)
    _daxpy_(a4[1]*h[0],k2,tmp)
    _daxpy_(a4[2]*h[0],k3,tmp)
    _daxpy_(a4[3]*h[0],k4,tmp)
    _daxpy_(1.,x,tmp)


    f(t[0]+c[3]*h[0], tmp, k5)

    tmp = dbl_vec(ndim)
    _daxpy_(a5[0]*h[0],k1,tmp)
    _daxpy_(a5[1]*h[0],k2,tmp)
    _daxpy_(a5[2]*h[0],k3,tmp)
    _daxpy_(a5[3]*h[0],k4,tmp)
    _daxpy_(a5[4]*h[0],k5,tmp)
    _daxpy_(1.,x,tmp)

    f(t[0]+c[4]*h[0], tmp, k6)

    tmp = dbl_vec(ndim)
    _daxpy_(a6[0]*h[0],k1,tmp)
    _daxpy_(a6[1]*h[0],k2,tmp)
    _daxpy_(a6[2]*h[0],k3,tmp)
    _daxpy_(a6[3]*h[0],k4,tmp)
    _daxpy_(a6[4]*h[0],k5,tmp)
    _daxpy_(a6[5]*h[0],k6,tmp)

    f(t[0]+c[5]*h[0], tmp, k7)


    x5 = dbl_vec(ndim)
    x4 = dbl_vec(ndim)

    tmp = dbl_vec(ndim)
    _daxpy_(b4[0],k1,tmp)
    _daxpy_(b4[1],k2,tmp)
    _daxpy_(b4[2],k3,tmp)
    _daxpy_(b4[3],k4,tmp)
    _daxpy_(b4[4],k5,tmp)
    _daxpy_(b4[5],k6,tmp)
    _daxpy_(b4[6],k7,tmp)
    _daxpy_(h[0],tmp,x4)


    tmp = dbl_vec(ndim)
    _daxpy_(b5[0],k1,tmp)
    _daxpy_(b5[1],k2,tmp)
    _daxpy_(b5[2],k3,tmp)
    _daxpy_(b5[3],k4,tmp)
    _daxpy_(b5[4],k5,tmp)
    _daxpy_(b5[5],k6,tmp)
    _daxpy_(b5[6],k7,tmp)
    _daxpy_(h[0],tmp,x5)

    q = 4.
    err = 0.

    for i in range(ndim):
        if cy_fabs(x4[i]) > cy_fabs(x5[i]):
            sc = atol + cy_fabs(x4[i])*rtol
        else:
            sc = atol + cy_fabs(x5[i])*rtol
        err_tmp = cy_fabs(x4[i]-x5[i])/sc
        if err_tmp > err:
            err = err_tmp


    if err == 0:
        h_opt = cy_copysign(10, h[0])
    else:
        h_opt = h[0]*cy_pow(1./err, 1./(q+1.))
    if err > 1:
        h[0] = fac * h_opt
    else:
        t[0] = t[0] + h[0]
        _daxpy_(1.,x5,x)
        if maxfac*h[0] > fac*h_opt:
            h[0] = maxfac*h[0]
        else:
            h[0] = fac*h_opt



@cython.cdivision(True)
cdef void func(double t, dbl_vec x, dbl_vec &ret) nogil:
    cdef:
        int i

    #assert ret.size() == x.size()
    ret[0] = x[0]
    ret[1] = 1./(1+cy_pow(t,2))
    ret[2] = x[2]

@cython.cdivision(True)
cdef void _estimate_e_and_pi_(double h, double *r):
    cdef:
        double t0, tf
        double t
        dbl_vec ret
        int i, niter
    ret = dbl_vec(3)
    t0 = 0.
    tf = 1.

    ret[0] = 1
    ret[2] = 1

#    niter = int(cy_ceil((tf-t0)/h))

    t = t0

    while t < tf:
        if h > tf - t:
            h = tf - t
        rkdp54_p(&t, ret, &h, func)

#    for i in range(niter):
#        rk3_p(&t, ret, &h, func)

    r[0] = ret[0]
    r[1] = 4*ret[1]
    r[2] = ret[2]






def estimate_e_and_pi(double h):
    cdef:
        double r[3]
    _estimate_e_and_pi_(h, &r[0])
    return r[0], r[1], r[2]
