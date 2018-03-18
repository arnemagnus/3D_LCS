import numpy as np
import argparse
from mpi4py import MPI
import numba as nb
import os
import sys
import errno

from numerical_integrators.singlestep import euler, rk2, rk3, rk4
from numerical_integrators.adaptive_step import rkbs32, rkbs54, rkdp54, rkdp87

def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

class RHS:
    def __init__(self, aperiodic):
        self.k0 = 0.3
        self.k1 = 0.5
        self.k2 = 1.5
        self.k3 = 1.8
        self.A = np.sqrt(3)
        self.B = np.sqrt(2)
        self.C = 1
        self._a = lambda t: self.A

        if aperiodic:
            self._b = lambda t: self.B*(1 + self.k0 * np.tanh(self.k1*t)*np.cos((self.k2*t)**2))
            self._c = lambda t: self.C*(1 + self.k0 * np.tanh(self.k1*t)*np.sin((self.k3*t)**2))
        else:
            self._b = lambda t: self.B
            self._c = lambda t: self.C

    def __call__(self, t, x):
        ret = np.empty(x.shape)
        ret[0] = self._a(t)*np.sin(x[2]) + self._c(t)*np.cos(x[1])                # x-component of velocity field
        ret[1] = self._b(t)*np.sin(x[0]) + self._a(t)*np.cos(x[2])                # y-component of velocity field
        ret[2] = self._c(t)*np.sin(x[1]) + self._b(t)*np.cos(x[0])                # z-component of velocity field
        ret[3] = -self._c(t)*np.sin(x[1])*x[6] + self._a(t)*np.cos(x[2])*x[9]     # The remaining (coupled) entries
        ret[4] = -self._c(t)*np.sin(x[1])*x[7] + self._a(t)*np.cos(x[2])*x[10]    # constitute the RHS of the
        ret[5] = -self._c(t)*np.sin(x[1])*x[8] + self._a(t)*np.cos(x[2])*x[11]    # variational ODE for the
        ret[6] = self._b(t)*np.cos(x[0])*x[3] + -self._c(t)*np.sin(x[2])*x[9]     # flow map Jacobian
        ret[7] = self._b(t)*np.cos(x[0])*x[4] + -self._c(t)*np.sin(x[2])*x[10]
        ret[8] = self._b(t)*np.cos(x[0])*x[5] + -self._c(t)*np.sin(x[2])*x[11]
        ret[9] = -self._b(t)*np.sin(x[0])*x[3] + self._c(t)*np.cos(x[1])*x[6]
        ret[10] = -self._b(t)*np.sin(x[0])*x[4] + self._c(t)*np.cos(x[1])*x[7]
        ret[11] = -self._b(t)*np.sin(x[0])*x[5] + self._c(t)*np.cos(x[1])*x[8]
        return ret

@nb.njit
def f(t,x,A=_a, B=_b,C=_c):
    """A function which computes the right hand side(s) of the coupled
    equation of variations for the flow map Jacobian, for (un)steady ABC flow.

    param: t -- Time
    param: x -- Twelve-component (NumPy) array, containing the flow map and
		Jacobian at time t. Shape: (12,nx,ny,nz)
    OPTIONAL:
    param: A -- Time-aperiodic (for unsteady flow) function handle for the
		A-amplitude. Default: A(t) = sqrt(3)
    param: B -- Time-aperiodic (for unsteady flow) function handle for the
		B-amplitude. Default: B(t) = sqrt(2)
    param: C -- Time-aperiodic (for unsteady flow) function handle for the
                C-amplitude. Default: C(t) = 1

    return: Twelve-component array, containing the (component-wise) right hand
	    side of the coupled equation of variations. Shape: (12,nx,ny,nz)

    """
    ret = np.empty(x.shape)
    ret[0] = A(t)*np.sin(x[2]) + C(t)*np.cos(x[1])                # x-component of velocity field
    ret[1] = B(t)*np.sin(x[0]) + A(t)*np.cos(x[2])                # y-component of velocity field
    ret[2] = C(t)*np.sin(x[1]) + B(t)*np.cos(x[0])                # z-component of velocity field
    ret[3] = -C(t)*np.sin(x[1])*x[6] + A(t)*np.cos(x[2])*x[9]     # The remaining (coupled) entries
    ret[4] = -C(t)*np.sin(x[1])*x[7] + A(t)*np.cos(x[2])*x[10]    # constitute the RHS of the
    ret[5] = -C(t)*np.sin(x[1])*x[8] + A(t)*np.cos(x[2])*x[11]    # variational ODE for the
    ret[6] = B(t)*np.cos(x[0])*x[3] + -A(t)*np.sin(x[2])*x[9]     # flow map Jacobian
    ret[7] = B(t)*np.cos(x[0])*x[4] + -A(t)*np.sin(x[2])*x[10]
    ret[8] = B(t)*np.cos(x[0])*x[5] + -A(t)*np.sin(x[2])*x[11]
    ret[9] = -B(t)*np.sin(x[0])*x[3] + C(t)*np.cos(x[1])*x[6]
    ret[10] = -B(t)*np.sin(x[0])*x[4] + C(t)*np.cos(x[1])*x[7]
    ret[11] = -B(t)*np.sin(x[0])*x[5] + C(t)*np.cos(x[1])*x[8]
    return ret

def generate_grid_and_save_eigenvariables(t0, tf, h, nx, ny, nz,
                                          integrator, fixed_step_integrators,
                                          atol, rtol, aperiodic):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    xmin, xmax = 0, 2*np.pi
    ymin, ymax = 0, 2*np.pi
    zmin, zmax = 0, 2*np.pi

    x = np.linspace(xmin,xmax,nx)
    y = np.linspace(ymin,ymax,ny)
    z = np.linspace(zmin,zmax,nz)
    f = RHS(aperiodic)
    print('Grid generated in rank {} of {}.'.format(rank,size))
    jac = find_endpoint_jacobian(t0,x,y,z,tf,h,f,integrator,fixed_step_integrators,atol,rtol)
    print('Endpoint Jacobian slice identified in rank {} of {}.'.format(rank,size))
    if rank == 0:
        u, s, v = np.linalg.svd(jac)
        print('SVD decomposition completed')

        lm1 = s[...,2]**2
        lm2 = s[...,1]**2
        lm3 = s[...,0]**2

        xi1 = v[...,2]
        xi2 = v[...,1]
        xi3 = v[...,0]

        print('Eigenvalues and -vectors extracted')

        config = '{}_t0={}_tf={}_nx={}_ny={}_nz={}_h={}_atol={}_rtol={}_aperiodic={}'.format(integrator.__name__,t0,tf,nx,ny,nz,h,atol,rtol,aperiodic)
        path = 'precomputed_strain_params/'+config+'/'
        ensure_path_exists(path)
        print('Wrapping folder generated')
        np.save(path+'lm1.npy',np.ascontiguousarray(lm1))
        np.save(path+'lm2.npy',np.ascontiguousarray(lm2))
        np.save(path+'lm3.npy',np.ascontiguousarray(lm3))
        np.save(path+'xi1.npy',np.ascontiguousarray(xi1))
        np.save(path+'xi2.npy',np.ascontiguousarray(xi2))
        np.save(path+'xi3.npy',np.ascontiguousarray(xi3))
        print('Strain eigenvalues and -vectors stored')
        np.save(path+'x.npy',x)
        np.save(path+'y.npy',y)
        np.save(path+'z.npy',z)
        print('Grid abscissae stored')


def find_endpoint_jacobian(t0,x,y,z,tf,h,func,integ,fixed_step_integrators,atol,rtol):
    """A function which computes the final state of the flow map Jacobian for three-dimensional
    tracer advection.

    param: t0 -- Start time
    param: x  -- (NumPy) array of abscissae values along the x-axis, i.e.,
                 the x-values of the points in the computational grid.
    param: y  -- (NumPy) array of abscissae values along the y-axis, i.e.,
                 the y-values of the points in the computational grid.
    param: z  -- (NumPy) array of abscissae values along the z-axis, i.e.,
                 the z-values of the points in the computational grid.
    param: tf -- End time
    param: h  -- (Initial) integration time step
    param: func -- Function handle, pointing to function returning the RHS
                   of the flow ODE system
    param: integ -- Function handle, pointing to function which performs
                    numerical integration (e.g., a Runge-Kutta solver)
    param: fixed_step_integrators -- Set of function handles pointing to
                                     the available Runge-Kutta solvers of
                                     fixed stepsize
    param: atol -- Absolute tolerance level (for adaptive solvers)
    param: rtol -- Relative tolerance level (for adaptive solvers)

    return: jac -- (NumPy) array of final state Jacobian values, with shape
                   (nx,ny,nz,3,3)
    """

    grid = np.zeros((12,x.shape[0],y.shape[0],z.shape[0]))
    grid[:3] = np.array(np.meshgrid(x,y,z,indexing='ij'))    # Initial conditions (to be advected)
    grid[3]  = 1                                             # Initial condition for the flow map Jacobian:
    grid[7]  = 1                                             #     Identity matrices
    grid[11] = 1

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()

    div = np.floor(np.linspace(0,x.shape[0],size+1)).astype(int)

    recv_buf = [np.empty((grid.shape[0],div[j+1]-div[j],grid.shape[2],grid.shape[3])) for j in range(1,size)]

    if rank == 0:
        grid[:,div[rank]:div[rank+1]] = _advect_slice(t0,grid[:,div[rank]:div[rank+1]],tf,h,func,integ,fixed_step_integrators,atol,rtol)
        print('Rank 0 advection done')
        for j in range(1,size):
            comm.Recv(recv_buf[j-1],j)
            grid[:,div[j]:div[j+1]] = recv_buf[j-1]
        return grid[3:].transpose(1,2,3,0).reshape((x.shape[0],y.shape[0],z.shape[0],3,3))
    else:
        comm.Send(_advect_slice(t0,grid[:,div[rank]:div[rank+1]],tf,h,func,integ,fixed_step_integrators,atol,rtol),dest=0)




def _advect_slice(t0,pos,tf,h,func,integ,fixed_step_integrators,atol,rtol):
    """A function which advects a slice of initial conditions from the initial
    to the final state.

    param: t0 -- Start time
    param: pos -- (NumPy) array of initial conditions.
    param: tf -- End time
    param: h -- (Initial) integration step
    param: func -- Function handle, pointing to a function returning
                   the RHS of the flow ODE system
    param: integ -- Function handle, pointing to function which performs
                    numerical integration (e.g., a Runge-Kutta solver)
    param: fixed_step_integrators -- Set of function handles pointing to
                                     the available Runge-Kutta solvers of
                                     fixed stepsize
    param: atol -- Absolute tolerance level (for adaptive solvers)
    param: rtol -- Relative tolerance level (for adaptive solvers)

    return: pos -- (NumPy) array of advected final state
    """
    if integ in fixed_step_integrators:
        t = t0
        for j in range(np.ceil((tf-t0)/h).astype(int)):
            t,pos,h = integ(t,pos,h,func)
        return pos
    else:
        t = np.ones(pos.shape[1:])*t0
        h = np.ones(pos.shape[1:])*h
        while np.any(np.less(t,tf)):
            h = np.minimum(h,tf-t)
            t,pos,h = integ(t,pos,h,func,atol,rtol)
        return pos

if __name__ == '__main__':
    fixed_step_integrators = set([euler, rk2, rk3, rk4])
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', help = 'integrator name. Default: rkdp54.', type = str, default='rkdp54')
    parser.add_argument('--stride', help='(initial) integration step. Default: 0.01.', type = float, default=0.01)
    parser.add_argument('--atol', help = 'absolute tolerance (adaptive methods). Default: 1e-5.', type = float, default=1.e-5)
    parser.add_argument('--rtol', help = 'relative tolerance (adaptive methods). Default: 1e-7.', type = float, default=1.e-7)
    parser.add_argument('--t0', help = 'start time. Default: 0.', type = float, default= 0)
    parser.add_argument('--tf', help = 'end time. Default: 2.', type = float, default = 2)
    parser.add_argument('--nx', help = 'number of points along x abscissa. Default: 101.', type = int, default=101)
    parser.add_argument('--ny', help = 'number of points along y abscissa. Default: 102.', type = int, default=102)
    parser.add_argument('--nz', help = 'number of points along z abscissa. Default: 103.', type = int, default=103)
    parser.add_argument('--aperiodic', help = 'whether or not to use aperiodic ABC flow. Default: False', type = bool, default = False)
    args = parser.parse_args()

    integrators = [euler, rk2, rk3, rk4, rkbs32, rkbs54, rkdp54, rkdp87]

    integrator = None

    for integ in integrators:
        if integ.__name__ == args.method:
            integrator = integ
            break

    if integrator is None:
        msg = 'Invalid choice of flow map integrator.'
        msg += '\n\t\tAvailable choices:'
        for i in integrators:
            msg += '\n' + '\t\t\t' + i.__name__
        raise RuntimeError(msg)
    print('Arguments parsed')
    generate_grid_and_save_eigenvariables(args.t0, args.tf, args.stride,
                                          args.nx, args.ny, args.nz,
                                          integrator, fixed_step_integrators,
                                          args.atol, args.rtol, args.aperiodic)


