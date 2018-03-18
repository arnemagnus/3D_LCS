"""
This module contains a variety of numerical integration schemes,
including both single-, multi- and adaptive timestep methods.

The various integrators have different stability properties and
accuracy levels, which means that they are suitable for problems
of different complexities and required levels of precisions.

With the single- and multistep methods, the price one pays for
increasing precision is, put simply, more floating-point
operations. This increases computation time, and increases the
numerical errors in the proposed solutions.

Adaptive timestep methods, on the other hand, can be trickier to
parallellize than their fixed timestep siblings. This is because
two different trajectories are generally traversed with two
different time steps.

All the numerical integrators have the same structure:

def scheme(t, x, h, f, atol, rtol):
    [...]
    return _t, _x, _h

where   t:    Current time level
        x:    Current coordinates, array-like
        h:    Current time increment (fixed for fixed stepsize methods,
                generally variable in adaptive stepsize methods)
        f:    Function handle for the derivatives (the RHS of the ODE system),
                function signature: f = f(t, x)
        atol: Absolute tolerance level (OPTIONAL, AND THAT ONLY FOR ADAPTIVE
                STEPSIZE METHODS)
        rtol: Relative tolerance level (OPTIONAL, AND THAT ONLY FOR ADAPTIVE
                STEPSIZE METHODS)

        _t:   New time level (fixed stepsize integrators always take one step
                forwards, whereas adaptive stepsize integrators do not if the
                trial solution is rejected, returning instead the current time
                level, unaltered)
        _x:   Approximation of the coordinates at the new time level
                (always the case for fixed stepsize integrators, not the case
                for addaptive stepsize integrators if the trial solution is
                rejected, returning instead the current coordinates, unaltered)
        _h:   New time increment (unaltered in fixed stepsize methods,
                generally increased or decreased in adaptive stepsize methods)
"""

# Changelog:
#     2017-09-19: File created
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.

import numerical_integrators.adaptive_step
import numerical_integrators.multistep
import numerical_integrators.singlestep
