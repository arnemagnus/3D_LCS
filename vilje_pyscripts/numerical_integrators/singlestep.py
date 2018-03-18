"""
This module contains a selection of fixed increment singlestep integrators
intended for general-purpose use. All integrators have the same function
signature, as in, they take the same input parameters and return the same
output variables in the same order, with the difference being the underlying
integration scheme.

The module contains the following fixed increment single-step
integrators:
    euler: Euler scheme (1st order)
    rk2:   Heun scheme (2nd order)
    rk3:   Kutta scheme (3rd order)
    rk4:   Runge-Kutta scheme (4th order)

All functions have the same structure:

def scheme(t, x, h, f, **kwargs):
    [...]
    return _t, _x, _h

where
    t:        Current time level
    x:        Current coordinates, array-like
    h:        Time increment (fixed)
    f:        Function handle for the derivatives (the RHS of the ODE system),
                function signature: f = f(t, x)
    **kwargs: Keyword arguments for the derivatives (optional)

    _t:       New time level
    _x:       Approximation of the coordinates at the new time level
    _h:       Time increment (unaltered, yet returned, in order for the return
                variable signatures of the numerical integrators to remain
                consistent across single-, multi- and adaptive step methods.
"""

# Changelog:
#       2017-08-25: Explicit Euler,
#                   Heun,
#                   RK3,
#                   RK4
#                   methods successfully implemented
#                   for the first time, albeit with a radically
#                   different overhanging file structure.
#
#       2017-08-30: Function signature for the derivative function
#                   standardized clarified (via code comments)
#
#       2017-09-01: Changed function signature of the integrator as well
#                   as the derivative function, from f(x,t) --> f(t,x),
#                   in accordance with convention from literature.
#
#       2017-09-19: Radically altered the structure of the numerical
#                   integrator package. From here on out, each
#                   integrator kind is contained within its own module,
#                   facilitating finding any given integrator in the
#                   event that changes must be made.
#
#                   In addition, the integrators now follow a more
#                   logical hierarchial system, with single-step
#                   integrators clearly differentiated from their
#                   multi-step brethren, for instance.
#
#                   This change was partially made with multi-step
#                   methods in mind, where a single-step method
#                   must be used at the first step, but also as a means
#                   to provide more robust program code which should
#                   be easier to maintain than was the case for my
#                   original structure.
#
#       2017-10-04: Enabled optional keyword argument input, directed towards
#                   the derivative functions, for all integrators.
#
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.


#--------------------------------------------------------------------#
#                  The explicit Euler (RK1) method                   #
#--------------------------------------------------------------------#

def euler(t, x, h, f, **kwargs):
    """    This function performs a single time step forwards, using the
    explicit Euler scheme, finding an approximation of the coordinates
    at the new time level.

    The explicit Euler scheme is the simplest Runge-Kutta scheme,
    and is first-order accurate.

    Input:
       t:        Current time level
       x:        Current coordinates, array-like
       h:        Time increment (fixed)
       f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
       **kwargs: Keyword arguments for the derivatives (optional)

    Output:
       _t:       New time level
       _x:       Explicit Euler approximation of the coordinates at the
                      new time level
       _h:       Time increment (unaltered, yet returned, in order for
                      the return variable signatures of the numerical
                      integrators to remain consistent across single-,
                      multi- and adaptive step methods
    """
    # Find "slope"
    k1 = f(t, x, **kwargs)
    # Find new time level
    _t = t + h
    # Find estimate of coordinates at new time level
    _x = x + k1*h

    return _t, _x, h



#--------------------------------------------------------------------#
#                  The explicit Heun (RK2) method                    #
#--------------------------------------------------------------------#

def rk2(t, x, h, f, **kwargs):
    """    This function performs a single time step forwards, using the
    Heun scheme (also known as the RK2 scheme), finding an
    approximation of the coordinates at the new time level.

    The Heun scheme is a member of the Runge-Kutta family of ODE
    solvers, and is second-order accurate.

    Input:
       t:        Current time level
       x:        Current coordinates, array-like
       h:        Time increment (fixed)
       f:        Function handle for the derivatives (the RHS of the ODE
                     system), function signature: f = f(t, x)
       **kwargs: Keyword arguments for the derivatives (optional)

    Output:
       _t:       New time level
       _x:       Heun approximation of the coordinates at the new time
                    level
       _h:        Time increment (unaltered, yet returned, in order for
                     the return variable signatures of the numerical
                     integrators to remain consistent across single-,
                     multi- and adaptive step methods
    """

    # Find "slopes"
    k1 = f(t    , x         , **kwargs)
    k2 = f(t + h, x + k1 * h, **kwargs)
    # Find new time level
    _t = t + h
    # Find estimate for coordinates at new time level
    _x = x + (k1 + k2)*h/2

    return _t, _x, h



#--------------------------------------------------------------------#
#                  The explicit Kutta (RK3) method                   #
#--------------------------------------------------------------------#

def rk3(t, x, h, f, **kwargs):
    """    This function performs a single time step forwards, using the
    Kutta (RK3) scheme, finding an approximation of the coordinates at
    the new time level.

    The Kutta scheme is a member of the Runge-Kutta family of ODE
    solvers, and is third-order accurate.

    Input:
       t:        Current time level
       x:        Current coordinates, array-like
       h:        Time increment (fixed)
       f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
       **kwargs: Keyword arguments for the derivatives (optional)

    Output:
       _t:        New time level
       _x:        Kutta approximation of the coordinates at the
                     new time level
       _h:        Time increment (unaltered, yet returned, in order for
                      the return variable signatures of the numerical
                      integrators to remain consistent across single-,
                      multi- and adaptive step methods
    """
    # Find "slopes"
    k1 = f(t      , x                , **kwargs)
    k2 = f(t + h/2, x + k1*h/2       , **kwargs)
    k3 = f(t + h  , x - k1*h + 2*k2*h, **kwargs)
    # Find new time level
    _t = t + h
    # Find estimate for coordinates at new time level
    _x = x + (k1 + 4*k2 + k3)*h/6

    return _t, _x, h



#--------------------------------------------------------------------#
#                  "The" explicit Runge-Kutta (RK4) method           #
#--------------------------------------------------------------------#

def rk4(t, x, h, f, **kwargs):
    """    This function performs a single time step forwards, using the
    classical Runge-Kutta (RK4) scheme, finding an approximation of the
    coordinates at the new time level.

    The classical Runge-Kutta scheme is a member of the Runge-Kutta
    family of ODE solvers, and is fourth-order accurate.

    Input:
       t:        Current time level
       x:        Current coordinates, array-like
       h:        Time increment (fixed)
       f:        Function handle for the derivatives (the RHS of the ODE
                     system), function signature: f = f(t, x)
       **kwargs: Keyword arguments for the derivatives (optional)

    Output:
       _t:       New time level
       _x:       Runge-Kutta approximation of the coordinates at the
                     new time level
       _h:       Time increment (unaltered, yet returned, in order for
                      the return variable signatures of the numerical
                      integrators to remain consistent across single-,
                      multi- and adaptive step methods
    """
    # Find "slopes"
    k1 = f(t      , x         , **kwargs)
    k2 = f(t + h/2, x + k1*h/2, **kwargs)
    k3 = f(t + h/2, x + k2*h/2, **kwargs)
    k4 = f(t + h  , x + k3*h  , **kwargs)
    # Find new time level
    _t = t + h
    # Find estimate for coordinates at new time level
    _x = x + (k1 + 2*k2 + 2*k3 + k4)*h/6

    return _t, _x, h
