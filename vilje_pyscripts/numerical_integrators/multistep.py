"""
This module contains a selection of multistep integrators intended for general-
purpose use. All integrators have the same function signature, as in, they take
the same input parameters and return the same output variables in the same
order, with the difference being the underlying integration scheme.

The module contains the following adaptive step size integrators:

where the digit outside of the parenthesis indicates the method order, and the
digit within the parenthesis indicates the order of the interpolant solution
(used in adjusting the time step).

All functions have the same structure:

def scheme(t, x, h, f):
   [...]
   return _t, _x, _h

where   t:    Current time level
        x:    Current coordinates, array-like
        h:    Current time increment
        f:    Function handle for the derivatives (the RHS of the ODE system),
                function signature: f = f(t, x)

        _t:   New time level (if trial step is accepted)
                Current time level (unaltered, if the trial step is rejected)
        _x:   Approximation of the coordinates at the new time level
                (if trial step is accepted)
                Current coordinates (unaltered, if the trial step is rejected)
        _h:   Updated time increment. Generally increased or decreased,
                depending on whether the trial step is accepted or not
"""


# Changelog:
#     2017-09-19: Module created
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.


#--------------------------------------------------------------------#
#                  The Adams-Bashford method                         #
#--------------------------------------------------------------------#

class AdaBas:

    def __init__(self, t, x, h, f):
        self.t = t

        self.x = x

        self.h = h
        self.f = f



