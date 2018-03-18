"""
This module contains a selection of adaptive timestep integrators intended for
general-purpose use. All integrators have the same function signature, as in,
they take the same input parameters and return the same output variables in the
same order, with the difference being the underlying integration scheme.

The module contains the following adaptive step size integrators:
    rkbs32: Bogacki-Shampine 3(2)
    rkbs54: Bogacki-Shampine 5(4)
    rkck45: Cash-Karp 4(5)
    rkdp54: Dormand-Prince 5(4)
    rkdp87: Dormand-Prince 8(7)
    rkf12:  Runge-Kutta-Fehlberg 1(2)
    rkf45:  Runge-Kutta-Fehlberg 4(5)
    rkf78:  Runge-Kutta-Fehlberg 7(8)
    rkhe21: Heun-Euler 2(1)

where the digit outside of the parenthesis indicates the method order, and the
digit within the parenthesis indicates the order of the interpolant solution
(used in adjusting the time step).

All functions have the same structure:

def scheme(t, x, h, f, atol, rtol, **kwargs):
   [...]
   return _t, _x, _h

where   t:        NumPy array containing the current time level(s)
        x:        Numpy array condaining the current coordinates
        h:        NumPy array containing the current time increment(s)
        f:        Function handle for the derivatives (the RHS of the ODE
                    system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative tolerance level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

        _t:       NumPy array containing
                    a) New time level (if trial step is accepted)
                    b) Current time level (unaltered, if the trial step is
                        rejected)
        _x:       NumPy array containing
                    a) Approximation of the coordinates at the new time level
                        (if trial step is accepted)
                    b) Current coordinates (unaltered, if the trial step is
                        rejected)
        _h:       NumPy array containing the updated time increment.
                    Generally increased or decreased,
                    depending on whether the trial step is accepted or not
"""


# Changelog:
#       2017-09-01: Dormand-Prince 5(4) method successfully implemented
#                   for the first time, albeit with a radically
#                   different overhanging file structure.
#
#       2017-09-07: Bogacki-Shampine 3(2),
#                   Bogacki-Shampine 5(4),
#                   Cash-Karp 4(5),
#                   Dormand-Prince 8(7) and
#                   Fehlberg 5(4)
#                   methods successfully implemented for the first time,
#                   albeit with a radically different overhanging file
#                   structure.
#
#       2017-09-19: Module created.
#                   Radically altered the structure of the numerical
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
#                   Fehlberg 1(2) method and Heun-Euler 2(1) methods
#                   implemented.
#
#       2017-09-27: Changed the timestep refinement in the Heun-Euler 2(1)
#                   integrator, such that it handles calculating several
#                   trajectories in parallel, each with their own timestep.
#
#                   The practical consequence is that it now only works
#                   for numpy array input, even in the case of a single
#                   trajectory, where the input time must be cast as
#                   t = np.array([t]), and similarly for the input timestep.
#
#       2017-10-02: Made the same changes mentioned above for the Heun-Euler 2(1)
#                   scheme, to all the other adaptive timestep integrators. All
#                   of them are now able to calculate several trajectories in
#                   parallel, e.g. by means of array slicing.
#
#                   Changed the Dormand-Prince 8(7) algorithm, such that the
#                   coefficients are given with 20 decimal precision, rather
#                   than the quite ridiculously long fractions, as was the
#                   case previously.
#
#       2017-10-04: Enabled optional keyword argument input, directed towards
#                   the derivative functions, for all integrators.
#
#       2017-10-16: Added Fehlberg 7(8) method.
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.

#--------------------------------------------------------------------#
#                      Common default parameters                     #
#--------------------------------------------------------------------#
atol_default = 1.e-6
rtol_default = 1.e-9

fac     = 0.8
maxfac  = 2

# numpy contains very useful representations of abs(:) and
# max(:) functions, among other things:
import numpy as np


#--------------------------------------------------------------------#
#                  The Bogacki-Shampine 3(2) method                  #
#--------------------------------------------------------------------#

def rkbs32(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Bogacki-Shampine 3(2) adaptive timestep integrator scheme. If
    the new step is not accepted, the time level and the coordinates
    are not updated, while the time increment is refined.

    The Bogacki-Shampine 3(2) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    second and third order, respectively. The scheme is tuned such
    that the error of the third order solution is minimal.

    The second order solution (interpolant) is used in order to find
    a criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
            some threshold, the solution is rejected, and the time
            increment refined
        - If the difference between the solutions is smaller than or
            equal to some threshold, the third order solution is
            accepted, and the solver attempts to increase the time
            increment


    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)
    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                           rejected)
        _x:       NumPy array containing
                      a) Bogacki-Shampine 3(2) approximation of the coordinates
                           at the new time level (if the trial step is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                           rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected
    """

    # Nodes
    c2 = 1./2.
    c3 = 3./4.
    c4 = 1.

    # Matrix elements
    a21 = 1./2.
    a31 = 0.
    a32 = 3./4.
    a41 = 2./9.
    a42 = 1./3.
    a43 = 4./9.

    # Second order weights
    b21 = 7./24.
    b22 = 1./4.
    b23 = 1./3.
    b24 = 1./8.

    # Third order weights
    b31 = 2./9.
    b32 = 1./3.
    b33 = 4./9.
    b34 = 0.

    # Find "slopes"
    k1 = f(t       , x                                 , **kwargs)
    k2 = f(t + c2*h, x + a21*h*k1                      , **kwargs)
    k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2           , **kwargs)
    k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3, **kwargs)

    # Find second and third order prediction of new point
    x_2 = x + h*(b21*k1 + b22*k2 + b23*k3 + b24*k4)
    x_3 = x + h*(b31*k1 + b32*k2 + b33*k3 + b34*k4)

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 3rd order, with 2nd order interpolation, hence:
    q = 2.

    sc = atol + np.maximum(np.abs(x_2), np.abs(x_3)) * rtol
    err = np.amax(np.sqrt((x_2-x_3)**2)/sc)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                                (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))


    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))] = \
                                        x_3[np.array([accepted_mask,]*len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask], \
                                       fac * h_opt[accepted_mask] \
                                      )

    # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(np.greater(err, 1.)):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                                            x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h



#--------------------------------------------------------------------#
#                  The Bogacki-Shampine 5(4) method                  #
#--------------------------------------------------------------------#


def rkbs54(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Bogacki-Shampine 5(4) adaptive timestep integrator scheme. If the
    new step is not accepted, the time level and the coordinates are
    not updated, while the time increment is refined.

    The Bogacki-Shampine 5(4) method calculates three independent
    approximations to a step forwards in time for an ODE system, one
    fifth and two of fourth order, respectively. The scheme is tuned
    tuned such that the error of the fifth order solution is minimal.

    The fourth order solutions (interpolant) are used in order to find
    a criterion for rejecting / accepting the trial step:
        - If the difference between the fifth order solution and either
          of the fourth order solutions is larger than some threshold
          the solution is rejected, and the time increment refined
        - If the difference between fifth order solution and both the
          fourth order solutions is smaller than or equal to some
          some threshold, the fifth order solution is accepted, and the
          solver attempts to increase the time increment


    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)

        **kwargs: Keyword arguments for the derivatives (optional)

    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                     rejected)
        _x:       NumPy array containing
                      a) Bogacki-Shampine 5(4) approximation of the coordinates
                          at the new time level (if the trial step is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected

    """

    # Nodes
    c2 = 1./6.
    c3 = 2./9.
    c4 = 3./7.
    c5 = 2./3.
    c6 = 3./4.
    c7 = 1.
    c8 = 1.

    # Matrix elements
    a21 = 1./6.
    a31 = 2./27.
    a32 = 4./27.
    a41 = 183./1372.
    a42 = -162./343.
    a43 = 1053./1372.
    a51 = 68./297.
    a52 = -4./11.
    a53 = 42./143.
    a54 = 1960./3861.
    a61 = 597./22528.
    a62 = 81./352.
    a63 = 63099./585728.
    a64 = 58653./366080.
    a65 = 4617./20480.
    a71 = 174197./959244.
    a72 = -30942./79937.
    a73 = 8152137./19744439.
    a74 = 666106./1039181.
    a75 = -29421./29068.
    a76 = 482048./414219.
    a81 = 587./8064.
    a82 = 0.
    a83 = 4440339./15491840.
    a84 = 24353./124800.
    a85 = 387./44800.
    a86 = 2152./5985.
    a87 = 7267./94080.

    # First of the fourth-order weights
    b41 = 6059./80640.
    b42 = 0.
    b43 = 8559189./30983680.
    b44 = 26411./124800.
    b45 = -927./89600.
    b46 = 443./1197.
    b47 = 7267./94080.
    b48 = 0.

    # Second of the fourth-order weights
    _b41 = 2479./34992.
    _b42 = 0.
    _b43 = 123./416.
    _b44 = 612941./3411720.
    _b45 = 43./1440.
    _b46 = 2272./6561.
    _b47 = 79937./1113912.
    _b48 = 3293./556956.

    # Fifth-order weights
    b51 = 587./8064.
    b52 = 0.
    b53 = 4440339./15491840.
    b54 = 24353./124800.
    b55 = 387./44800.
    b56 = 2152./5985.
    b57 = 7267./94080.
    b58 = 0.

    # Find "slopes"
    k1 = f(t       , x                                              , **kwargs)
    k2 = f(t + c2*h, x + a21*h*k1                                   , **kwargs)
    k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                        , **kwargs)
    k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3             , **kwargs)
    k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4  , **kwargs)
    k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4
            + a65*h*k5                      , **kwargs)
    k7 = f(t + c7*h, x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4
            + a75*h*k5 + a76*h*k6           , **kwargs)
    k8 = f(t + c8*h, x + a81*h*k1 + a82*h*k2 + a83*h*k3 + a84*h*k4
            + a85*h*k5 + a86*h*k6 + a87*h*k7, **kwargs)

    # Find fourth and fifth order prediction of new point
    x_4 = x + h*( b41*k1 +  b42*k2 +  b43*k3 +  b44*k4 +  b45*k5
            +  b46*k6 +  b47*k7 +  b48*k8)
    _x_4 = x + h*(_b41*k1 + _b42*k2 + _b43*k3 + _b44*k4 + _b45*k5
            + _b46*k6 + _b47*k7 + _b48*k8)
    x_5 = x + h*( b51*k1 +  b52*k2 +  b53*k3 +  b54*k4+  b55*k5
            +  b56*k6 +  b57*k7 +  b58*k8)

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 5th order, with 4th order interpolation, hence:
    q = 4.

    sc = atol + np.maximum(np.abs(x_4), np.abs(_x_4)) * rtol
    err = np.amax(np.sqrt((x_4-_x_4)**2)/sc)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Trajectories which fail to satisfy our tolerance restrictions at the first
    # trial step are not updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        h_opt[rejected_mask] = h[rejected_mask] * \
                (1./err[rejected_mask]) ** (1./(q + 1.))
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    # For trajectories where the first trial step is accepted:
    accepted_first_mask = np.less_equal(err, 1)
    if np.any(accepted_first_mask):
        # Moving forwards, we only need the trajectories which pass the first
        # trial step, hence:
        _x_4 = _x_4[np.array([accepted_first_mask,]*len(x))]
        x_5 = x_5[np.array([accepted_first_mask,]*len(x))]
        h_opt = h_opt[accepted_first_mask]


        sc = atol + np.maximum(np.abs(_x_4), np.abs(x_5)) * rtol
        err = np.amax(np.sqrt((_x_4-x_5)**2)/sc)


        # Should the error happen to be zero, the optimal timestep is infinity.
        # We set an upper limit in order to ensure sensible behaviour.
        # In addition, we make sure we step in the direction originally intended;
        # when integrating backwards in time, we need negative timesteps, hence:
        if np.any(np.equal(err, 0.)):
            h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

        # For nonzero error, the calculation is fairly straightforward:
        if np.any(np.greater(err, 0.)):
            h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                    (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))

        # If any trajectories satisfy our tolerance restrictions, the
        # corresponding time levels, positions and timesteps are updated:
        accepted_second_mask = np.less_equal(err, 1)
        if np.any(accepted_second_mask):
            _t[accepted_first_mask] += (t[accepted_first_mask] + \
                    h[accepted_first_mask]\
                    ) * accepted_second_mask
            _x[np.array([accepted_first_mask,]*len(x))] += \
                    x_5 * accepted_second_mask
            _h[accepted_first_mask] += np.maximum(\
                    maxfac * h[accepted_first_mask],\
                    fac * h_opt * accepted_second_mask\
                    )

            # If any trajectories fail the second trial step, the corresponding
        # time level(s) and coordinates are not updated, while the timestep
        # is decreased:
        rejected_second_mask = np.greater(err, 1.)
        if np.any(rejected_second_mask):
            _t[accepted_first_mask] += t[accepted_first_mask]
            _x[np.array([accepted_first_mask,]*len(x))] += \
                    x[accepted_first_mask] * rejected_second_mask
            _h[accepted_first_mask] = fac * h_opt * rejected_second_mask



    return _t, _x, _h



#--------------------------------------------------------------------#
#                  The Cash-Karp 4(5) method                         #
#--------------------------------------------------------------------#

def rkck45(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Cash-Karp 4(5) adaptive timestep integrator scheme. If the
    new step is not accepted, the time level and the coordinates are
    not updated, while the time increment is refined.

    The Cash-Karp 4(5) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    fourth and fifth order, respectively. The scheme is tuned such
    that the error of the fourth order solution is minimal.

    The fourth order solution (interpolant) is used in order to find
    a criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
          some threshold, the solution is rejected, and the time
          increment refined
        - If the difference between the solutions is smaller than or
          equal to some threshold, the fourth order solution is
          accepted, and the solver attempts to increase the time
          increment

    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                          rejected)
        _x:       NumPy array containing
                      a) Cash-Karp 4(5) approximation of the coordinates at
                          the new time level (if the trial step is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                     Generally increased or decreased,
                     depending on whether the trial step is accepted or
                     rejected

    """

    # Nodes
    c2 = 1./5.
    c3 = 3./10.
    c4 = 3./5.
    c5 = 1.
    c6 = 7./8.

    # Matrix elements
    a21 = 1./5.
    a31 = 3./40.
    a32 = 9./40.
    a41 = 3./10.
    a42 = -9./10.
    a43 = 6./5.
    a51 = -11./54.
    a52 = 5./2.
    a53 = -70./27.
    a54 = 35./27.
    a61 = 1631./55296.
    a62 = 175./512.
    a63 = 575./13824.
    a64 = 44275./110592.
    a65 = 253./4096.

    # Fourth-order weights
    b41 = 2825./27648.
    b42 = 0.
    b43 = 18575./48384.
    b44 = 13525./55296.
    b45 = 277./14336.
    b46 = 1./4.

    # Fifth-order weights
    b51 = 37./378.
    b52 = 0.
    b53 = 250./621.
    b54 = 125./594.
    b55 = 0.
    b56 = 512./1771.

    # Find "slopes"
    k1 = f(t       , x                                               , **kwargs)
    k2 = f(t + c2*h, x + a21*h*k1                                    , **kwargs)
    k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                         , **kwargs)
    k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3              , **kwargs)
    k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4   , **kwargs)
    k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4
            + a65*h*k5, **kwargs)

    # Find fourth and fifth order prediction of new point
    x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6)
    x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6)

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 4th order, with 5th order interpolation, hence:
    q = 5.

    sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
    err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    if np.any(np.equal(err, 0.)):
        h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    if np.any(np.greater(err, 0.)):
        h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))


    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))]= x_4[np.array([accepted_mask,]\
                *len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask], \
                fac * h_opt[accepted_mask])

        # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                x[np.array([np.greater(err, 1.),]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h



#--------------------------------------------------------------------#
#                  The Dormand-Prince 5(4) method                    #
#--------------------------------------------------------------------#

def rkdp54(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Dormand-Prince 5(4) adaptive timestep integrator scheme. If the
    new step is not accepted, the time level and the coordinates are
    not updated, while the time increment is refined.

    The Dormand-Prince 5(4) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    fifth and fourth order, respectively. The scheme is tuned such that
    the error of the fifth order solution is minimal.

    The fourth order solution (interpolant) is used in order to find a
    criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
          some threshold, the solution is rejected, and the time
          increment refined
       - If the difference between the solutions is smaller than or
          equal to some threshold, the fifth order solution is
          accepted, and the solver attempts to increase the time
          increment

    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

   Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                          rejected)
        _x:       NumPy array containing
                      a) Dormand-Prince 5(4) approximation of the coordinates
                          at the new time level (if the trial step is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected
   """

    # Nodes
    c2 = 1./5.
    c3 = 3./10.
    c4 = 4./5.
    c5 = 8./9.
    c6 = 1.
    c7 = 1.

    # Matrix elements
    a21 = 1./5.
    a31 = 3./40.
    a32 = 9./40.
    a41 = 44./45.
    a42 = -56./15.
    a43 = 32./9.
    a51 = 19372./6561.
    a52 = -25360./2187.
    a53 = 64448./6561.
    a54 = -212./729.
    a61 = 9017./3168.
    a62 = -355./33.
    a63 = 46732./5247.
    a64 = 49./176.
    a65 = -5103./18656.
    a71 = 35./384.
    a72 = 0.
    a73 = 500./1113.
    a74 = 125./192.
    a75 = -2187./6784.
    a76 = 11./84.

    # Fourth-order weights
    b41 = 5179./57600.
    b42 = 0.
    b43 = 7571./16695.
    b44 = 393./640.
    b45 = -92097./339200.
    b46 = 187./2100.
    b47 = 1./40.

    # Fifth-order weights
    b51 = 35./384.
    b52 = 0.
    b53 = 500./1113.
    b54 = 125./192.
    b55 = -2187./6784.
    b56 = 11./84.
    b57 = 0.

    # Find "slopes"
    k1 = f(t       , x                                               , **kwargs)
    k2 = f(t + c2*h, x + a21*h*k1                                    , **kwargs)
    k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                         , **kwargs)
    k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3              , **kwargs)
    k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4   , **kwargs)
    k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4
                                                + a65*h*k5           , **kwargs)
    k7 = f(t + c7*h, x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4
                                                + a75*h*k5 + a76*h*k6, **kwargs)

    # Find fourth and fifth order prediction of new point
    x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6 + b47*k7)
    x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6 + b57*k7)

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 5th order, with 4th order interpolation, hence:
    q = 4.

    sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
    err = np.amax(np.sqrt((x_4-x_5)**2)/sc,axis=0)

    print(err.shape)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    if np.any(np.equal(err, 0.)):
        h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    if np.any(np.greater(err, 0.)):
        h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                                  (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))


    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))] = \
                                          x_5[np.array([accepted_mask,]*len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask],\
                                       fac * h_opt[accepted_mask]\
                                      )

    # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                                            x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h




#--------------------------------------------------------------------#
#                  The Dormand-Prince 8(7) method                    #
#--------------------------------------------------------------------#

def rkdp87(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Dormand-Prince 8(7) adaptive timestep integrator scheme. If the
    new step is not accepted, the time level and the coordinates are
    not updated, while the time increment is refined.

    The Dormand-Prince 8(7) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    eighth and seventh order, respectively. The scheme is tuned such
    that the error of the eighth order solution is minimal.

    The seventh order solution (interpolant) is used in order to find
    a criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
          some threshold, the solution is rejected, and the time
          increment refined
        - If the difference between the solutions is smaller than or
          equal to some threshold, the eighth order solution is
          accepted, and the solver attempts to increase the time
          increment

    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                          rejected)
        _x:       NumPy array containing
                      a) Dormand-Prince 8(7) approximation of the coordinates
                          at the new time level (if the trial step is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected

    """

    # The nodes, matrix elements and weights are here truncated at their
    # twentieth decimal value, exceeding the default Python machine epsilon
    # by about five orders of magnitude. The exact numerical values consist
    # of fractions, where several contain more than 100 digits in both
    # the numerator and the denominator, rendering the code quite unreadable.

    # Nodes
    c2  = 0.05555555555555555556
    c3  = 0.08333333333333333333
    c4  = 0.125
    c5  = 0.3125
    c6  = 0.375
    c7  = 0.1475
    c8  = 0.465
    c9  = 0.56486545138225957539
    c10 = 0.65
    c11 = 0.92465627764050444674
    c12 = 1.
    c13 = 1.

    # Matrix elements
    a21   = 0.055555555555555556
    a31   = 0.020833333333333333
    a32   = 0.0625
    a41   = 0.03125
    a42   = 0
    a43   = 0.09375
    a51   = 0.3125
    a52   = 0.
    a53   = -1.171875
    a54   = 1.17185
    a61   = 0.0375
    a62   = 0.
    a63   = 0.
    a64   = 0.1875
    a65   = 0.15
    a71   = 0.047910137111111111
    a72   = 0.
    a73   = 0.
    a74   = 0.112248712777777777
    a75   = -0.025505673777777777
    a76   = 0.012846823888888888
    a81   = 0.016917989787292281
    a82   = 0.
    a83   = 0.
    a84   = 0.387848278486043169
    a85   = 0.035977369851500327
    a86   = 0.196970214215666060
    a87   = -0.172713852340501838
    a91   = 0.069095753359192300
    a92   = 0.
    a93   = 0.
    a94   = -0.63424797672885411
    a95   = -0.16119757522460408
    a96   = 0.138650309458825255
    a97   = 0.940928614035756269
    a98   = 0.211636326481943981
    a101  = 0.183556996839045385
    a102  = 0.
    a103  = 0.
    a104  = -2.46876808431559245
    a105  = -0.29128688781630045
    a106  = -0.02647302023311737
    a107  = 2.847838764192800449
    a108  = 0.281387331469849792
    a109  = 0.123744899863314657
    a111  = -1.21542481739588805
    a112  = 0.
    a113  = 0.
    a114  = 16.672608665945772432
    a115  = 0.915741828416817960
    a116  = -6.05660580435747094
    a117  = -16.00357359415617811
    a118  = 14.849303086297662557
    a119  = -13.371575735289849318
    a1110 = 5.134182648179637933
    a121  = 0.258860916438264283
    a122  = 0.
    a123  = 0.
    a124  = -4.774485785489205112
    a125  = -0.435093013777032509
    a126  = -3.049483332072241509
    a127  = 5.577920039936099117
    a128  = 6.155831589861040689
    a129  = -5.062104586736938370
    a1210 = 2.193926173180679061
    a1211 = 0.134627998659334941
    a131  = 0.822427599626507477
    a132  = 0.
    a133  = 0.
    a134  = -11.658673257277664283
    a135  = -0.757622116690936195
    a136  = 0.713973588159581527
    a137  = 12.075774986890056739
    a138  = -2.127659113920402656
    a139  = 1.990166207048955418
    a1310 = -0.234286471544040292
    a1311 = 0.175898577707942265
    a1312 = 0.

    # Seventh-order weights
    b71  = 0.0295532136763534969
    b72  = 0.
    b73  = 0.
    b74  = 0.
    b75  = 0.
    b76  = -0.8286062764877970397
    b77  = 0.3112409000511183279
    b78  = 2.4673451905998869819
    b79  = -2.5469416518419087391
    b710 = 1.4435485836767752403
    b711 = 0.0794155958811272872
    b712 = 0.0444444444444444444
    b713 = 0.

    # Eighth-order weights
    b81  = 0.0417474911415302462
    b82  = 0.
    b83  = 0.
    b84  = 0.
    b85  = 0.
    b86  = -0.0554523286112393089
    b87  = 0.2393128072011800970
    b88  = 0.7035106694034430230
    b89  = -0.7597596138144609298
    b810 = 0.6605630309222863414
    b811 = 0.1581874825101233355
    b812 = -0.2381095387528628044
    b813 = 0.25


    # Find "slopes"
    k1  = f(t         , x                                            , **kwargs)
    k2  = f(t +  c2*h , x +  a21*h*k1                                , **kwargs)
    k3  = f(t +  c3*h , x +  a31*h*k1 +  a32*h*k2                    , **kwargs)
    k4  = f(t +  c4*h , x +  a41*h*k1 +  a42*h*k2 +  a43*h*k3        , **kwargs)
    k5  = f(t +  c5*h , x +  a51*h*k1 +  a52*h*k2 +  a53*h*k3
                          +  a54*h*k4                                , **kwargs)
    k6  = f(t +  c6*h , x +  a61*h*k1 +  a62*h*k2 +  a63*h*k3
                          +  a64*h*k4 +  a65*h*k5                    , **kwargs)
    k7  = f(t +  c7*h , x +  a71*h*k1 +  a72*h*k2 +  a73*h*k3
                          +  a74*h*k4 +  a75*h*k5 +  a76*h*k6        , **kwargs)
    k8  = f(t +  c8*h , x +  a81*h*k1 +  a82*h*k2 +  a83*h*k3
                        +  a84*h*k4 +  a85*h*k5 +  a86*h*k6
                           +  a87*h*k7                               , **kwargs)
    k9  = f(t +  c9*h , x +  a91*h*k1 +  a92*h*k2 +  a93*h*k3
                        +  a94*h*k4  +  a95*h*k5 +  a96*h*k6
                           +  a97*h*k7 +  a98*h*k8                   , **kwargs)
    k10 = f(t + c10*h, x + a101*h*k1 + a102*h*k2 + a103*h*k3
                        + a104*h*k4 + a105*h*k5 + a106*h*k6
                           + a107*h*k7 + a108*h*k8 + a109*h*k9       , **kwargs)
    k11 = f(t + c11*h, x + a111*h*k1 + a112*h*k2 + a113*h*k3
                        + a114*h*k4 + a115*h*k5 + a116*h*k6
                        + a117*h*k7 + a118*h*k8 + a119*h*k9
                            + a1110*h*k10                            , **kwargs)
    k12 = f(t + c12*h, x + a121*h*k1 + a122*h*k2 + a123*h*k3
                        + a124*h*k4 + a125*h*k5 + a126*h*k6
                        + a127*h*k7 + a128*h*k8 + a129*h*k9
                            + a1210*h*k10 + a1211*h*k11              , **kwargs)
    k13 = f(t + c13*h, x + a131*h*k1 + a132*h*k2 + a133*h*k3
                            + a134*h*k4 + a135*h*k5 + a136*h*k6
                            + a137*h*k7 + a138*h*k8 + a139*h*k9
                            + a1310*h*k10 + a1311*h*k11 + a1312*h*k12, **kwargs)

    # Find seventh and eighth order prediction of new point
    x_7 = x + h*( b71*k1 +  b72*k2 +  b73*k3 +  b74*k4 +  b75*k5
            +  b76*k6 +  b77*k7 +  b78*k8 +  b79*k9
            + b710*k10 + b711*k11 + b712*k12 + b713*k13  )
    x_8 = x + h*( b81*k1 +  b82*k2 +  b83*k3 +  b84*k4 +  b85*k5
            +  b86*k6 +  b87*k7 +  b88*k8 +  b89*k9
            + b810*k10 + b811*k11 + b812*k12 + b813*k13  )

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 8th order, with 7th order interpolation, hence:
    q = 7.

    sc = atol + np.maximum(np.abs(x_7), np.abs(x_8)) * rtol
    err = np.amax(np.sqrt((x_7-x_8)**2)/sc)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    if np.any(np.equal(err, 0.)):
        h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    if np.any(np.greater(err, 0.)):
        h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                                (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))


    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))] = \
                                        x_8[np.array([accepted_mask,]*len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask], \
                                       fac * h_opt[accepted_mask] \
                                      )

    # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                                            x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h



#--------------------------------------------------------------------#
#                  The Fehlberg 1(2) method                          #
#--------------------------------------------------------------------#

def rkf12(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Fehlberg 1(2) adaptive timestep integrator scheme. If the
    new step is not accepted, the time level and the coordinates are
    not updated, while the time increment is refined.

    The Fehlberg 1(2) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    first and second order, respectively. The scheme is tuned such that
    the error of the first order solution is minimal.

    The second order solution (interpolant) is used in order to find a
    criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
          some threshold, the solution is rejected, and the time
          increment refined
        - If the difference between the solutions is smaller than or
          equal to some threshold, the first order solution is
          accepted, and the solver attempts to increase the time
          increment

    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                          rejected)
        _x:       NumPy array containing
                      a) Runge-Kutta-Fehlberg 4(5) approximation of the
                          coordinates at the new time level (if the trial step
                          is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected
    """

    # Nodes
    c2 = 1./2.
    c3 = 1.

    # Matrix elements
    a21 = 1./2.
    a31 = 1./256.
    a32 = 255./256.

    # First-order weights
    b11 = 1./256,
    b12 = 255./256.
    b13 = 0.

    # Second-order weights
    b21 = 1./512.
    b22 = 255./256.
    b23 = 1./512.

    # Find "slopes"
    k1 = f(t       , x                      , **kwargs)
    k2 = f(t + c2*h, x + h*a21*k1           , **kwargs)
    k3 = f(t + c3*h, x + h*a31*k1 + h*a32*k2, **kwargs)

    # Find first and second order prediction of new point
    x_1 = x + h*(b11*k1 + b12*k2 + b13*k3)
    x_2 = x + h*(b21*k1 + b22*k2 + b23*k3)

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 1st order, with 2nd order interpolation, hence:
    q = 2.

    sc = atol + np.maximum(np.abs(x_1), np.abs(x_2)) * rtol
    err = np.amax(np.sqrt((x_1-x_2)**2)/sc)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    if np.any(np.equal(err, 0.)):
        h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    if np.any(np.greater(err, 0.)):
        h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                                (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))


    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))] = \
                                        x_1[np.array([accepted_mask,]*len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask], \
                                       fac * h_opt[accepted_mask] \
                                      )

    # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                                            x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h



#--------------------------------------------------------------------#
#                  The Fehlberg 4(5) method                          #
#--------------------------------------------------------------------#

def rkf45(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Runge-Kutta-Fehlberg 4(5) adaptive timestep integrator scheme. If
    the new step is not accepted, the time level and the coordinates
    are not updated, while the time increment is refined.

    The Runge-Kutta-Fehlberg 4(5) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    fifth and fourth order, respectively. The scheme is tuned such
    that the error of the fourth order solution is minimal.

    The fifth order solution (interpolant) is used in order to find
    a criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
          some threshold, the solution is rejected, and the time
          increment refined
        - If the difference between the solutions is smaller than or
          equal to some threshold, the fourth order solution is
          accepted, and the solver attempts to increase the time
          increment

    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                          rejected)
        _x:       NumPy array containing
                      a) Runge-Kutta-Fehlberg 4(5) approximation of the
                          coordinates at the new time level (if the trial step
                          is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected
    """

    # Nodes
    c2 = 1./4.
    c3 = 3./8.
    c4 = 12./13.
    c5 = 1.
    c6 = 1./2.

    # Matrix elements
    a21 = 1./4.
    a31 = 3./32.
    a32 = 9./32.
    a41 = 1932./2197.
    a42 = -7200./2197.
    a43 = 7296./2197.
    a51 = 439./216.
    a52 = -8.
    a53 = 3680./513
    a54 = -845./4104.
    a61 = -8./27.
    a62 = 2.
    a63 = -3544./2565.
    a64 = 1859./4104.
    a65 = -11./40.

    # Fourth-order weights
    b41 = 25./216.
    b42 = 0.
    b43 = 1408./2565.
    b44 = 2197./4104.
    b45 = -1./5.
    b46 = 0.

    # Fifth-order weights
    b51 = 16./135.
    b52 = 0.
    b53 = 6656./12825.
    b54 = 28561./56430.
    b55 = -9./50.
    b56 = 2./55.

    # Find "slopes"
    k1 = f(t       , x                                               , **kwargs)
    k2 = f(t + c2*h, x + a21*h*k1                                    , **kwargs)
    k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                         , **kwargs)
    k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3              , **kwargs)
    k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4   , **kwargs)
    k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4
                                                           + a65*h*k5, **kwargs)

    # Find fourth and fifth order prediction of new point
    x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6)
    x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6)

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 4th order, with 5th order interpolation, hence:
    q = 5.

    sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
    err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    if np.any(np.equal(err, 0.)):
        h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    if np.any(np.greater(err, 0.)):
        h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                                (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))


    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))] = \
                                        x_4[np.array([accepted_mask,]*len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask], \
                                       fac * h_opt[accepted_mask] \
                                      )

    # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                                            x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h



#--------------------------------------------------------------------#
#                  The Fehlberg 7(8) method                          #
#--------------------------------------------------------------------#

def rkf78(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Runge-Kutta-Fehlberg 7(8) adaptive timestep integrator scheme. If
    the new step is not accepted, the time level and the coordinates
    are not updated, while the time increment is refined.

    The Runge-Kutta-Fehlberg 7(8) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    eighth and seventh order, respectively. The scheme is tuned such
    that the error of the seventh order solution is minimal.

    The eigth order solution (interpolant) is used in order to find
    a criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
          some threshold, the solution is rejected, and the time
          increment refined
        - If the difference between the solutions is smaller than or
          equal to some threshold, the seventh order solution is
          accepted, and the solver attempts to increase the time
          increment

    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                          rejected)
        _x:       NumPy array containing
                      a) Runge-Kutta-Fehlberg 7(8) approximation of the
                          coordinates at the new time level (if the trial step
                          is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected
    """

    # Nodes
    c2  = 2./27.
    c3  = 1./9.
    c4  = 1./6.
    c5  = 5./12.
    c6  = 1./2.
    c7  = 5./6.
    c8  = 1./6.
    c9  = 2./3.
    c10 = 1./3.
    c11 = 1.
    c12 = 0.
    c13 = 1.

    # Matrix elements
    a21   = 2./27.
    a31   = 1./36.
    a32   = 1./12.
    a41   = 1./24.
    a42   = 0.
    a43   = 1./8.
    a51   = 5./12.
    a52   = 0.
    a53   = -25./16.
    a54   = 25./16.
    a61   = 1./20.
    a62   = 0.
    a63   = 0.
    a64   = 1./4.
    a65   = 1./5.
    a71   = -25./108.
    a72   = 0.
    a73   = 0.
    a74   = 125./108.
    a75   = -65./27.
    a76   = 125./54.
    a81   = 31./300.
    a82   = 0.
    a83   = 0.
    a84   = 0.
    a85   = 61./225.
    a86   = -2./9.
    a87   = 13./900.
    a91   = 2.
    a92   = 0.
    a93   = 0.
    a94   = -53./6.
    a95   = 704./45.
    a96   = -107./9.
    a97   = 67./90.
    a98   = 3.
    a101  = -91./108.
    a102  = 0.
    a103  = 0.
    a104  = 23./108.
    a105  = -976./135.
    a106  = 311./54.
    a107  = -19./60.
    a108  = 17./6.
    a109  = -1./12.
    a111  = 2383./4100.
    a112  = 0.
    a113  = 0.
    a114  = -341./164.
    a115  = 4496./1025.
    a116  = -301./82.
    a117  = 2133./4100.
    a118  = 45./82.
    a119  = 45./164.
    a1110 = 18./41.
    a121  = 3./205.
    a122  = 0.
    a123  = 0.
    a124  = 0.
    a125  = 0.
    a126  = -6./41.
    a127  = -3./205.
    a128  = -3./41.
    a129  = 3./41.
    a1210 = 6./41.
    a1211 = 0.
    a131  = -1777./4100.
    a132  = 0.
    a133  = 0.
    a134  = -341./164.
    a135  = 4496./1025.
    a136  = -289./82.
    a137  = 2193./4100.
    a138  = 51./82.
    a139  = 33./164.
    a1310 = 12./41.
    a1311 = 0.
    a1312 = 1.

    # Seventh-order weights
    b71  = 41./840.
    b72  = 0.
    b73  = 0.
    b74  = 0.
    b75  = 0.
    b76  = 34./105.
    b77  = 9./35.
    b78  = 9./35.
    b79  = 9./280.
    b710 = 9./280.
    b711 = 41./840.
    b712 = 0.
    b713 = 0.

    # Eighth-order weights
    b81  = 0.
    b82  = 0.
    b83  = 0.
    b84  = 0.
    b85  = 0.
    b86  = 34./105.
    b87  = 9./35.
    b88  = 9./35.
    b89  = 9./280.
    b810 = 9./280.
    b811 = 0.
    b812 = 41./840.
    b813 = 41./840.

    # Find "slopes"
    k1  = f(t        , x                                            , **kwargs)

    k2  = f(t +  c2*h, x +    a21*h*k1                              , **kwargs)

    k3  = f(t +  c3*h, x +    a31*h*k1 +  a32*h*k2                  , **kwargs)

    k4  = f(t +  c4*h, x +    a41*h*k1 +  a42*h*k2 +  a43*h*k3      , **kwargs)

    k5  = f(t +  c5*h, x +    a51*h*k1 +  a52*h*k2 +  a53*h*k3
                         +    a54*h*k4                              , **kwargs)

    k6  = f(t +  c6*h, x +    a61*h*k1 +  a62*h*k2 +  a63*h*k3
                         +    a64*h*k4 +  a65*h*k5                  , **kwargs)

    k7  = f(t +  c7*h, x +    a71*h*k1 +  a72*h*k2 +  a73*h*k3
                         +    a74*h*k4 +  a75*h*k5 +  a76*h*k6      , **kwargs)

    k8  = f(t +  c8*h, x +    a81*h*k1 +  a82*h*k2 +  a83*h*k3
                         +    a84*h*k4 +  a85*h*k5 +  a86*h*k6
                         +    a87*h*k7                              , **kwargs)

    k9  = f(t +  c9*h, x +    a91*h*k1 +  a92*h*k2 +  a93*h*k3
                         +    a94*h*k4 +  a95*h*k5 +  a96*h*k6
                         +    a97*h*k7 +  a98*h*k8                  , **kwargs)

    k10 = f(t + c10*h, x +   a101*h*k1 + a102*h*k2 + a103*h*k3
                         +   a104*h*k4 + a105*h*k5 + a106*h*k6
                         +   a107*h*k7 + a108*h*k8 + a109*h*k9      , **kwargs)

    k11 = f(t + c11*h, x +   a111*h*k1 + a112*h*k2 + a113*h*k3
                         +   a114*h*k4 + a115*h*k5 + a116*h*k6
                         +   a117*h*k7 + a118*h*k8 + a119*h*k9
                         + a1110*h*k10                              , **kwargs)

    k12 = f(t + c12*h, x +   a121*h*k1 +   a122*h*k2 + a123*h*k3
                         +   a124*h*k4 +   a125*h*k5 + a126*h*k6
                         +   a127*h*k7 +   a128*h*k8 + a129*h*k9
                         + a1210*h*k10 + a1211*h*k11                , **kwargs)

    k13 = f(t + c13*h, x +   a131*h*k1 +   a132*h*k2 +   a133*h*k3
                         +   a134*h*k4 +   a135*h*k5 +   a136*h*k6
                         +   a137*h*k7 +   a138*h*k8 +   a139*h*k9
                         + a1310*h*k10 + a1311*h*k11 + a1312*h*k12  , **kwargs)

    # Find seventh and eighth order prediction of new point
    x_7 = x + h*( b71*k1 +   b72*k2 +   b73*k3 +   b74*k4 +   b75*k5
              +   b76*k6 +   b77*k7 +   b78*k8 +   b79*k9
              + b710*k10 + b711*k11 + b712*k12 + b713*k13  )
    x_8 = x + h*( b81*k1 +   b82*k2 +   b83*k3 +   b84*k4 +   b85*k5
             +   b86*k6  +   b87*k7 +   b88*k8 +   b89*k9
             + b810*k10  + b811*k11 + b812*k12 + b813*k13  )

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 7th order, with 8th order interpolation, hence:
    q = 8.

    sc = atol + np.maximum(np.abs(x_7), np.abs(x_8)) * rtol
    err = np.amax(np.sqrt((x_7-x_8)**2)/sc)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    if np.any(np.equal(err, 0.)):
        h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    if np.any(np.greater(err, 0.)):
        h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                                (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))


    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))] = \
                                        x_7[np.array([accepted_mask,]*len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask], \
                                       fac * h_opt[accepted_mask] \
                                      )

    # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                                            x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h



#--------------------------------------------------------------------#
#                  The Heun-Euler 2(1) method                        #
#--------------------------------------------------------------------#

def rkhe21(t, x, h, f, atol=atol_default, rtol = rtol_default, **kwargs):
    """
    This function attempts a single time step forwards, using the
    Heun-Euler 2(1) adaptive timestep integrator scheme. If the
    new step is not accepted, the time level and the coordinates are
    not updated, while the time increment is refined.

    The Heun-Euler 2(1) method calculates two independent
    approximations to a step forwards in time for an ODE system, of
    first and second order, respectively. The scheme is tuned such that
    the error of the second order solution is minimal.

    The first order solution (interpolant) is used in order to find a
    criterion for rejecting / accepting the trial step:
        - If the difference between the two solutions is larger than
          some threshold, the solution is rejected, and the time
          increment refined
        - If the difference between the solutions is smaller than or
          equal to some threshold, the second order solution is
          accepted, and the solver attempts to increase the time
          increment

    Input:
        t:        Current time level, as a NumPy array
        x:        Current coordinates, as a NumPy array
        h:        Current time increment, as a NumPy array
        f:        Function handle for the derivatives (the RHS of the ODE
                      system), function signature: f = f(t, x)
        atol:     Absolute tolerance level (OPTIONAL)
        rtol:     Relative toleranve level (OPTIONAL)
        **kwargs: Keyword arguments for the derivatives (optional)

    Output:
        _t:       NumPy array containing
                      a) New time level (if the trial step is accepted)
                      b) Current time level (unaltered, if the trial step is
                    rejected)
        _x:       NumPy array containing
                      a) Heun-Euler 2(1) approximation of the coordinates at
                          the new time level (if the trial step is accepted)
                      b) Current coordinates (unaltered, if the trial step is
                          rejected)
        _h:       NumPy array containing the updated time increment.
                      Generally increased or decreased,
                      depending on whether the trial step is accepted or
                      rejected
    """

    # Nodes
    c2 = 1.

    # Matrix elements
    a21 = 1.

    # First-order weights:
    b11 = 1.
    b12 = 0.

    # Second-order weights:
    b21 = 1./2.
    b22 = 1./2.

    # Find "slopes"
    k1 = f(t       , x           , **kwargs)
    k2 = f(t + c2*h, x + h*a21*k1, **kwargs)

    # Find first and second order prediction of new point
    x_1 = x + h*(b11*k1 + b12*k2)
    x_2 = x + h*(b21*k1 + b22*k2)

    # Implementing error check and variable stepsize roughly as in
    # Hairer, Nørsett and Wanner: "Solving ordinary differential
    #                              equations I -- Nonstiff problems",
    #                              pages 167 and 168 in the 2008 ed.

    # The method is 2nd order, with 1st order interpolation, hence:
    q = 1.

    sc = atol + np.maximum(np.abs(x_1), np.abs(x_2)) * rtol
    err = np.amax(np.sqrt((x_1-x_2)**2)/sc, axis = 0)

    # Preallocate arrays for the return variables, as well as the timestep
    # refinement:
    h_opt = np.zeros(np.shape(h))
    _t = np.zeros(np.shape(t))
    _x = np.zeros(np.shape(x))
    _h = np.zeros(np.shape(h))

    # Should the error happen to be zero, the optimal timestep is infinity.
    # We set an upper limit in order to ensure sensible behaviour.
    # In addition, we make sure we step in the direction originally intended;
    # when integrating backwards in time, we need negative timesteps, hence:
    if np.any(np.equal(err, 0.)):
        h_opt[np.equal(err, 0.)] = np.sign(h[np.equal(err, 0.)]) * 10

    # For nonzero error, the calculation is fairly straightforward:
    if np.any(np.greater(err, 0.)):
        h_opt[np.greater(err, 0.)] = h[np.greater(err, 0.)] * \
                                (1./err[np.greater(err, 0.)]) ** (1./(q + 1.))

    # If any trajectories satisfy our tolerance restrictions, the corresponding
    # time levels, positions and timesteps are updated:
    accepted_mask = np.less_equal(err, 1.)
    if np.any(accepted_mask):
        _t[accepted_mask] = t[accepted_mask] + h[accepted_mask]
        _x[np.array([accepted_mask,]*len(x))] = \
                                        x_2[np.array([accepted_mask,]*len(x))]
        _h[accepted_mask] = np.maximum(maxfac * h[accepted_mask], \
                                       fac * h_opt[accepted_mask] \
                                      )

    # Trajectories which fail to satisfy our tolerance restrictions are not
    # updated, and the timestep is decreased.
    rejected_mask = np.greater(err, 1.)
    if np.any(rejected_mask):
        _t[rejected_mask] = t[rejected_mask]
        _x[np.array([rejected_mask,]*len(x))] = \
                                            x[np.array([rejected_mask,]*len(x))]
        _h[rejected_mask] = fac * h_opt[rejected_mask]

    return _t, _x, _h
