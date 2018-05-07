import numpy as np
import time
#from mpi4py import MPI
import sys
import os
import errno
import argparse
import multiprocessing as mp
from ndcurvebsplineinterp import NDCurveBSplineInterpolator
#from trivariatevectorbsplineinterpolation import SplineAimAssister, SplineEigenvectorInterpolator, Dp87BSpline
#from trivariatevectorlinearinterpolation import LinearAimAssister, LinearEigenvectorInterpolator, Dp87Linear
#from trivariatescalarinterpolation import TrivariateSpline
from auxiliarypointstuff import cy_compute_pos_aim, cy_cross_product, \
                                cy_in_plane, cy_orthogonal_component, \
                                cy_parallel_component, cy_normalize, \
                                cy_norm2, cy_min, cy_max, cy_dot

def ensure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def compute_hessian_lm(lm,x,y,z):
    dx, dy, dz = x[1]-x[0], y[1]-y[0], z[1]-z[0]
    grad = np.gradient(lm,dx,dy,dz,edge_order=2)
    hessian = np.empty((x.shape[0],y.shape[0],z.shape[0],3,3))
    for k, grad_k in enumerate(grad):
        # Iterate over the number of dimensions
        # Apply gradient operator to each component of the first derivative
        tmp_grad = np.gradient(grad_k,dx,dy,dz,edge_order=2)
        for l, grad_kl in enumerate(tmp_grad):
            hessian[...,k,l] = grad_kl
    return hessian

def find_points_in_ab(hess_lm3,lm3,lm2,xi3,freq):
    mask_a = np.logical_and(np.greater(lm3,1),np.greater(lm3,lm2))
    mask_b = np.less_equal(np.sum(xi3*np.sum(hess_lm3*xi3[...,np.newaxis],axis=3),axis=3),0)
    mask_c = np.zeros(mask_a.shape,dtype=np.bool)
    mask_c[freq::freq,freq::freq,freq::freq] = True

    return np.logical_and(np.logical_and(mask_a,mask_b), mask_c)

def compute_manifold_initial_positions(hess_lm3,lm3,lm2,xi3,freq,x,y,z):
    mask = find_points_in_ab(hess_lm3,lm3,lm2,xi3,freq)
    initial_positions = []
    for i in range(lm3.shape[0]):
        for j in range(lm3.shape[1]):
            for k in range(lm3.shape[2]):
                if (mask[i,j,k]):
                    initial_positions.append(np.array([x[i],y[j],z[k]]))
    return initial_positions


def find_initial_positions(freq,rank):
    x, y, z = np.load('precomputed_strain_params/x.npy'),  np.load('precomputed_strain_params/y.npy'),  np.load('precomputed_strain_params/z.npy')
    lm2 = np.load('precomputed_strain_params/lm2.npy')
    lm3 = np.load('precomputed_strain_params/lm3.npy')
    xi3 = np.load('precomputed_strain_params/xi3.npy')
    hess_lm3 = compute_hessian_lm(lm3,x,y,z)
    initial_pos = compute_manifold_initial_positions(hess_lm3,lm3,lm2,xi3,freq,x,y,z)
    return initial_pos[rank]


class Manifold:
    """A wrapper class for a collection of geodesic level sets which
    constitute an invariant manifold.

    Methods defined here:

    Manifold.__init__(init_pos, dom_bound, max_geo_dist, dist, dist_tol,
                min_ang, max_ang, min_dist_ang, max_dist_ang,
                min_sep, max_sep, max_arclen_factor)
    Manifold.add_level_sets(num_sets_to_add)
    Manifold.check_ab()
    Manifold.compute_lambda3_and_weights()

    """

    def __init__(self, init_pos, dom_bound, max_geo_dist, dist, dist_tol,
                 min_ang, max_ang, min_dist_ang, max_dist_ang, min_sep, max_sep,
                 max_arclen_factor, init_num_points, init_radius, timelimit
                ):
        """Manifold.__init__(init_pos, dom_bound, max_geo_dist, dist, dist_tol,
                min_ang, max_ang, min_dist_ang, max_dist_ang,
                min_sep, max_sep, max_arclen_factor)

        Initializes a Manifold object without adding any level sets.

        param: init_pos --     NumPy array containing the initial position (x,y,z)
                               from which the manifold is constructed
        param: dom_bound --    Domain boundaries, as a six-element list.
                               Format: [xmin, xmax, ymin, ymax, zmin, zmax]
        param: max_geo_dist -- Maximum geodesic distance. Used to terminate
                               development of manifold
        param: dist --         The (initial) radial distance from each
                               point in a given level set, to the
                               'radially connected' point in the
                               construction of the next level set
        param: dist_tol --     Numerical tolerance parameter for the
                               above. 0 <= dist_tol <= 1
        param: min_ang --      Minimal radial angular deviation between
                               consecutive constructed level sets,
                               under which 'dist' is increased
                               before the _next_ level set is
                               constructed
        param: max_ang --      Maximal radial angular deviation between
                               consecutive constructed level sets,
                               over which 'dist' is decreased,
                               the most recent attempt at creating
                               a level set is discarded, and
                               attempted anew with decreased 'dist'
        param: min_dist_ang -- Minimal product of 'dist' and
                               radial angular deviation between
                               consecutive constructed level sets,
                               under which 'dist' is increased
                               before the _next_ level set is
                               constructed
        param: max_dist_ang -- Maximal product of 'dist' and
                               radial angular deviation between
                               consecutive constructed level sets,
                               over which 'dist' is decreased,
                               the most recent attempt at creating
                               a level set is discarded, and
                               attempted anew with decreased 'dist'
        param: min_sep --      Minimal distance allowed between
                               (neighboring) points in a level set.
        param: max_sep --      Maximal distance allowed between
                               (neighboring) points in a level set
        param: max_arclen_factor -- Scalar factor specifying for how
                                    long paths are integrated by
                                    RK solver relative to initial
                                    separation of source and target
        param: init_num_points --
        param: init_radius --
	param: timelimit    -- Time available for computation
        """
        self.levelsets    = []
        self.triangulations = []
        self.triangles    = []
        self.dist         = dist
        self.num_sets     = 0
        self.geo_dist     = 0
        self.index        = 1
        self.input_params = InputManifoldParameters(init_pos, dom_bound, max_geo_dist,
                                                     dist_tol, min_ang, max_ang, min_dist_ang,
                                                     max_dist_ang, min_sep, max_sep,
                                                     max_arclen_factor, init_num_points, init_radius
                                                   )
        self.set_num_triangles = []

        self.xs = np.asarray([self.input_params.init_pos[0]])
        self.ys = np.asarray([self.input_params.init_pos[1]])
        self.zs = np.asarray([self.input_params.init_pos[2]])

        self.consecutive_self_intersections = []


    def add_level_sets(self, num_sets_to_add):
        """Manifold.add_level_sets(num_sets_to_add)

        Adds a specified number of geodesic level sets to the Manifold.

        Writes the number of points in each successfully added level set,
        as well as the elapsed time, to console.

        If no more geodesic level sets can be added, exceptions with
        descriptive names and docstrings are raised.

        param: num_sets_to_add -- The (integer) number of geodesic level sets to add.

        """
        n = 0
        start_time = time.time()
        recent_times = []
        try:
            if self.num_sets == 0 and num_sets_to_add > 0:
                suggested_levelset = GeodesicLevelSet(self.num_sets, self.dist, self.index, self.geo_dist,
                                                       self.input_params)
                # No self-intersection test necessary
                self.levelsets.append(suggested_levelset)
                self.index = self.levelsets[-1].last_index
                for tri in self.levelsets[-1].triangulations:
                    self.triangulations.append(tri)
                for p in self.levelsets[-1].points:
                    self.xs = np.append(self.xs, p.pos[0])
                    self.ys = np.append(self.ys, p.pos[1])
                    self.zs = np.append(self.zs, p.pos[2])
                del self.levelsets[-1].triangulations, self.levelsets[-1].xs, self.levelsets[-1].ys, self.levelsets[-1].zs
                self.num_sets += 1
                n += 1
        except OutsideOfDomainError as e:
            print(e.value)
            print('Point near edge. Could not complete first level set. Returning empty manifold')
            return
        if len(self.consecutive_self_intersections) > 10:
            print('Self-intersection detected. No more levelsets can be added.')
            return
        while (n < num_sets_to_add and self.geo_dist <= self.input_params.max_geo_dist) and (time.time() - start_time + 2*sum(recent_times) < timelimit):
            t_start = time.time()
            try:
                suggested_levelset = GeodesicLevelSet(self.num_sets, self.dist, self.index, self.geo_dist,
                                                       self.input_params, self.levelsets[self.num_sets-1])
            except InsufficientPointsError as e:
                print('Insufficient amount of points in latest set to perform sensible triangulation.'\
                      'Disregarding latest suggestion, don''t fuck with squirrels, Morty!')
                break
            except (NeedSmallerDistError, OutsideOfDomainError) as e:
                if (type(e) is NeedSmallerDistError):
                    try:
                        if (self.dist > self.input_params.min_sep):
                            self.dist = self.input_params.min_sep
                            try:
                                suggested_levelset = GeodesicLevelSet(self.num_sets, self.dist, self.index,
                                                    self.geo_dist, self.input_params, self.levelsets[self.num_sets-1])
                            except NeedSmallerDistError as e:
                                print('Could not complete geodesic level set number {}'.format(n))
                                raise CannotDecreaseDistFurtherError('Cannot add more level sets without violating min_sep')
                        else:
                            raise CannotDecreaseDistFurtherError('Cannot add more level sets without violating min_sep')
                    except (OutsideOfDomainError, CannotDecreaseDistFurtherError) as e:
                        print(e.value)
                        break
                    except InsufficientPointsError as e:
                        print('Insufficient amount of points in latest set to perform sensible triangulation.'\
                              'Disregarding latest suggestion, don''t fuck with squirrels, Morty!')
                        break
                else:
                    print(e.value)
                    break

#            intersections, suggested_triangles = self.self_intersections(suggested_levelset)

            self.levelsets.append(suggested_levelset)
#            self.triangles = self.triangles + suggested_triangles

            self.triangulations = self.triangulations + self.levelsets[-1].triangulations

            self.xs = np.append(self.xs, suggested_levelset.xs)
            self.ys = np.append(self.ys, suggested_levelset.ys)
            self.zs = np.append(self.zs, suggested_levelset.zs)

            del self.levelsets[-1].triangulations, self.levelsets[-1].xs, self.levelsets[-1].ys, self.levelsets[-1].zs

            self.index = self.levelsets[-1].last_index
            self.num_sets += 1
            self.geo_dist += self.dist
            self.dist = self.levelsets[-1].next_dist
            n += 1
            print('Level set {:4d} completed. Number of points: {:4d}. Cumulative geodesic distance: {:.3f}.'\
                  ' Elapsed time: {:.2f} seconds.'.format(len(self.levelsets),
                                                        len(self.levelsets[-1].points),
                                                        self.geo_dist,
                                                        time.time() - t_start
                                                       )
                 )
#            if intersections:
#                self.consecutive_self_intersections.append(True)
#                if len(self.consecutive_self_intersections) > 10:
#                    print('Self-intersection detected. No more levelsets can be added.')
#                    break
#            else:
#                self.consecutive_self_intersections = []

            recent_times.append(time.time() - t_start)
            if (len(recent_times) > 5):
                recent_times.pop(0)


        if (self.geo_dist > self.input_params.max_geo_dist):
            print('Max geodesic distance reached. No more level sets can be added in the current environment.')


    def check_ab(self):
        """Manifold.check_ab()

        Checks which of the points in the parametrization of the manifold
        satisfy the A- and B- criteria for (strong) LCSs.

        The boolean flag 'in_ab' for each of the Point instances in all
        the GeodesicLevelSet instances is set when this function is
        called.
        """
        for level in self.levelsets:
            for point in level.points:
                if (point.in_ab is None):
                    point._is_in_ab()
        print('Points in AB domain identified.')

    def compute_lambda3_and_weights(self):
        """Manifold.compute_lambda3_and_weights()

        Assigns a B-spline interpolated lambda3 value to each point in the manifold,
        in addition to a numerical weighting factor intended to approximate the
        part of the surface area of the manifold which is associated to each
        individual point.
        """
        for i, lset in enumerate(self.levelsets):
            n = len(lset.points)
            for j, point in enumerate(lset.points):
                point.lambda3 = lm3_field(point.pos)
                point.weight = 0.5*(lset.dist + self.levelsets[min(i+1,len(self.levelsets)-1)].dist)*\
                               0.5*(cy_norm2(point.pos - lset.points[divmod(j-1,n)[1]].pos)+
                                    cy_norm2(point.pos - lset.points[divmod(j+1,n)[1]].pos)
                                   )

        print('Point weights identified.')

    def self_intersections(self, suggested_levelset):

        suggested_triangles = self.make_triangles(suggested_levelset)
        #print(sum(self.set_num_triangles[:-9]))
        #for t in self.triangles[:sum(self.set_num_triangles[:-9])]:
        #    print(t)
        for tri_lhs in self.triangles:#[:sum(self.set_num_triangles[:-9])]:
            for tri_rhs in suggested_triangles:
                if moller_trumbore_checker(tri_lhs,tri_rhs):
                #if (intersects(tri_lhs, tri_rhs)):
                    return True, suggested_triangles
        return False, suggested_triangles

    def make_triangles(self, geodesic_levelset):
        triangles = []
        vertices = np.empty((3,3))

        xs = np.append(self.xs, geodesic_levelset.xs)
        ys = np.append(self.ys, geodesic_levelset.ys)
        zs = np.append(self.zs, geodesic_levelset.zs)

        for tri in geodesic_levelset.triangulations:
            vertices[0,0] = xs[tri[0]]#geodesic_levelset.xs[tri[0]-self.xs.shape[0]]
            vertices[0,1] = ys[tri[0]]#geodesic_levelset.ys[tri[0]-self.xs.shape[0]]
            vertices[0,2] = zs[tri[0]]#geodesic_levelset.zs[tri[0]-self.xs.shape[0]]
            #print(vertices[0])
            #print(geodesic_levelset.xs[tri[0]-self.xs.shape[0]], geodesic_levelset.ys[tri[0]-self.xs.shape[0]],
            #     geodesic_levelset.zs[tri[0]-self.xs.shape[0]])

            vertices[1,0] = xs[tri[1]]#geodesic_levelset.xs[tri[1]-self.xs.shape[0]]
            vertices[1,1] = ys[tri[1]]#geodesic_levelset.ys[tri[1]-self.xs.shape[0]]
            vertices[1,2] = zs[tri[1]]#geodesic_levelset.zs[tri[1]-self.xs.shape[0]]

            vertices[2,0] = xs[tri[2]]#geodesic_levelset.xs[tri[2]-self.xs.shape[0]]
            vertices[2,1] = ys[tri[2]]#geodesic_levelset.ys[tri[2]-self.xs.shape[0]]
            vertices[2,2] = zs[tri[2]]#geodesic_levelset.zs[tri[2]-self.xs.shape[0]]

            triangles.append(Triangle3D(vertices))
        #self.set_num_triangles.append(len(triangles))
        return triangles

class InputManifoldParameters:
    """A wrapper class for a set of parameters which define an invariant manifold
    parametrized in terms of geodesic level sets.

    Methods defined here:

    InputManifoldParameters.__init__(init_pos, dom_bound, max_geo_dist, dist, dist_tol,
                                     min_ang, max_ang, min_dist_ang, max_dist_ang,
                                     min_sep, max_sep, max_arclen_factor
                                    )

    """

    # Constructor
    def __init__(self, init_pos, dom_bound, max_geo_dist, dist_tol,
                min_ang, max_ang, min_dist_ang, max_dist_ang, min_sep, max_sep,
                max_arclen_factor, init_num_points, init_radius):
        """InputManifoldParameters.__init__(init_pos, dom_bound, max_geo_dist, dist, dist_tol,
                                 min_ang, max_ang, min_dist_ang, max_dist_ang,
                                 min_sep, max_sep, max_arclen_factor
                                )

        For most practical purposes, a dictionary disguised as a class, intended for use in the interaction
        between a Manifold instance and its constituent GeodesicLevelSet instances, in order to shorten
        the involved call signatures.

        """

        self.init_pos = init_pos
        self.max_geo_dist = max_geo_dist
        self.dom_bound = dom_bound
        self.dist_tol = dist_tol
        self.min_ang = min_ang
        self.max_ang = max_ang
        self.min_dist_ang = min_dist_ang
        self.max_dist_ang = max_dist_ang
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.max_arclen_factor = max_arclen_factor
        self.init_num_points = init_num_points
        self.init_radius = init_radius

class GeodesicLevelSet:
    """A wrapper class for a collection of points which
    parametrize a geodesic level set.

    Methods defined here:

    GeodesicLevelSet.__init__(level_num, dist, input_params, prev_set)
    GeodesicLevelSet._generate_first_set(input_params)
    GeodesicLevelSet._generate_set(dist, input_params, prev_set)
    GeodesicLevelSet._revise_set(set_suggestion, prev_set, input_params)
    GeodesicLevelSet._stabs_in_the_dark(index, prev_set, input_params, inbetween)
    GeodesicLevelSet._remove_loops(set_suggestion, min_sep, max_sep)
    """


    # Constructor
    def __init__(self, level_num, dist, index, geo_dist, input_params, prev_set=None):
        """GeodesicLevelSet.__init__(level_num, dist, input_params, prev_set)

        Constructor for a GeodesicLevelSet instance.

        param: level_num --    The (integer) number of geodesic level sets which have
                               been computed prior to this one. Only used to differentiate
                               between the base-case construction routine for the very
                               first level set, which is made to be a small, planar
                               circle, and all others, which are constructed by means
                               of trajectori  es orthogonal to a vector field.
        param: dist --         If level_num == 0: Ten times the radius of the initial
                               level set, approximated as a planar circle.
                               Else: The (Euclidean) distance from each point in
                               the immediately preceding level set, at which
                               one wants to find a new level set.
        param: index --        The number of Point instances which have been
                               added to the overarching Manifold object.
                               Used in order to triangulate points for tri-surface
                               plotting.
        param: input_params -- An InputGeodesicParameters instance, containing
                               a set of parameters which define the parametrization
                               of the manifold, of which this GeodesicLevelSet is a
                               constituent part. See the InputGeodesicParameters
                               docstring for details.
        param: prev_set --     The GeodesicLevelSet instance representing the immediately
                               preceding geodesic level set in the manifold in question.
                               If None and level_num > 0: Raises an exception

        """

        self.dist = dist
        self.triangulations = []

        # Set in sub-functions: self.next_dist

        if (level_num == 0):
            new_set, index = self._generate_first_set(index, input_params)
        else:
            set_suggestion = self._generate_set(dist, input_params, prev_set)
            new_set, index = self._revise_set(set_suggestion, prev_set, index, geo_dist, input_params)

        self.points = new_set

        self.xs = np.asarray([p.pos[0] for p in self.points])
        self.ys = np.asarray([p.pos[1] for p in self.points])
        self.zs = np.asarray([p.pos[2] for p in self.points])


        self.next_dist = self.dist
        self.last_index = index

        # Lag spline-interpolasjon av punktene i levelsettet, hvor duplikat av første
        # punkt legges sist for å generere pseudo-periodisk interpolasjonsobjekt
        # (styres m/ wraparound = True), og vi 'padder' med et (likt) antall punkter
        # i begge retninger sett fra s = 0, for å skape en glattere skjøt.

        self.interpolation = NDCurveBSplineInterpolator(np.asarray([point.pos for point in self.points]),
                                                        wraparound=True,pad_points=2)

    def _generate_first_set(self, index, input_params):
        """GeodesicLevelSet._generate_first_set(input_params)

        Generates the initial geodesic level set in the parametrization of a
        manifold.

        *** This function is called by the constructor, explicitly calling this
            function should never be necessary and is not advised in general***
        param: index --        The number of Point instances which have been
                               added to the overarching Manifold object.
                               When this function is called, index should
                               always be 1 (as the Manifold 'origin' is
                               always added first). Used in order to triangulate
                               points for tri-surface plotting.
        param: input_params -- An InputGeodesicParameters instance, containing
                               a set of parameters which define the parametrization
                               of the manifold, of which this GeodesicLevelSet is a
                               constituent part. See the InputGeodesicParameters
                               docstring for details.

        return: first_set -- The first geodesic level set
        return: index -- The index of the last point in first_set

        """
        first_set = []

        xi3_init_pos = xi3_field(input_params.init_pos)

        if (xi3_init_pos[1] == 0 and xi3_init_pos[2] == 0):
            xi1_init_pos = np.array([0,1,0])
            xi2_init_pos = np.array([0,0,1])
        else:
            xi1_init_pos = cy_normalize(np.array([0,-xi3_init_pos[2], xi3_init_pos[1]]))
            xi2_init_pos = cy_normalize(np.cross(xi1_init_pos, xi3_init_pos))

        for i in range(input_params.init_num_points):
            newcoord = input_params.init_pos + input_params.init_radius*(xi1_init_pos
                        *np.cos(2*np.pi*i/input_params.init_num_points)
                        + xi2_init_pos*np.sin(2*np.pi*i/input_params.init_num_points))
            if (not in_domain(newcoord, input_params.dom_bound)):
                raise OutsideOfDomainError('Attempted to place point outside domain. Returning manifold')

            first_set.append(Point(pos=newcoord, prev_vec=cy_normalize(newcoord - input_params.init_pos),
                        tan_vec=cy_normalize(cy_cross_product(xi3_init_pos, cy_normalize(newcoord -
                        input_params.init_pos)))))


        # Setting indices
        for pnt in first_set:
            pnt.index = index
            index += 1

        self.next_dist = self.dist

        for i in range(1,len(first_set)):
            self.triangulations.append([0, i, i+1])
        self.triangulations.append([0,len(first_set),1])

        return first_set, index

    def _generate_set(self, dist, input_params, prev_set):
        """GeodesicLevelSet._generate_set_set(dist, input_params, prev_set)

        Generates a geodesic level set, for level_num > 0, in the parametrization
        of a manifold.

        *** This function is called by the constructor, explicitly calling this
            function should never be necessary and is not advised in general***

        param: dist --         The (Euclidean) distance from each point in
                               the immediately preceding level set, at which
                               one wants to find a new level set.
        param: input_params -- An InputGeodesicParameters instance, containing
                               a set of parameters which define the parametrization
                               of the manifold, of which this GeodesicLevelSet is a
                               constituent part. See the InputGeodesicParameters
                               docstring for details.
        param: prev_set --     The GeodesicLevelSet instance representing the immediately
                               preceding geodesic level set in the manifold in question.
                               If None: Raises an exception

        """
        set_suggestion = []
        if not prev_set:
            raise RuntimeError('Missing previous geodesic level set!')
        for i in range(len(prev_set.points)):
            set_suggestion.append(Point._find_ordinary_point(i, prev_set, input_params, dist, inbetween=False))
            if (not in_domain(set_suggestion[-1].pos, input_params.dom_bound)):
                raise OutsideOfDomainError('Attempted to place point outside domain. Returning manifold')


        return set_suggestion

    # Check that all restrictions are satisfied + setting next_dist
    def _revise_set(self, set_suggestion, prev_set, index, geo_dist, input_params):
        """GeodesicLevelSet._revise_set(set_suggestion, prev_set, input_params)

        A function which enforces a suggested geodesic level set to conform
        with preset tolerance levels (cf. input_params).

        *** This function is called by the constructor, explicitly calling this
            function should never be necessary and is not advised in general ***

        param: set_suggestion -- A suggestion for the _next_ geodesic level set,
                                 as computed by _generate_set
        param: prev_set       -- The immediately preceding geodesic level set
        param: index --          The number of Point instances which have been
                                 added to the overarching Manifold object.
                                 Used in order to triangulate points for tri-surface
                                 plotting.
        param:geo_dist --
        param: input_params   -- An InputGeodesicParameters instance, containing
                                 a set of parameters which define the parametrization
                                 of the manifold, of which this GeodesicLevelSet is a
                                 constituent part. See the InputGeodesicParameters
                                 docstring for details.

        return: set_suggestion -- (Usually) altered version of the input set_suggestion,
                                  where no points in the parametrization are too far
                                  apart or too close together, all points pass
                                  local curvature tests and detected loops resulting
                                  from numerical noise have been eliminated.
        return: index --          The index of the last point in set_suggestion

        """

        # Curvature tests
        over_max_ang, under_min_ang = curvature_test(set_suggestion, prev_set.points, input_params.min_ang,
                                                     input_params.max_ang)
        over_max_dist_ang, under_min_dist_ang = step_modified_curvature_test(set_suggestion, prev_set.points,
                                            self.dist, input_params.min_dist_ang, input_params.max_dist_ang)

        # If curvature is too large
        dist_reduced = False
        while ((over_max_ang or over_max_dist_ang) and self.dist >= 2*input_params.min_sep):
            self.dist = 0.5*self.dist
            set_suggestion = []
            for i in range(len(prev_set.points)):
                set_suggestion.append(Point._find_ordinary_point(i, prev_set, input_params, self.dist,
                                                             inbetween=False))
                if (not in_domain(set_suggestion[-1].pos, input_params.dom_bound)):
                    raise OutsideOfDomainError('Attempted to place point outside domain. Returning manifold')


            over_max_ang, under_min_ang = curvature_test(set_suggestion, prev_set.points, input_params.min_ang,
                                                    input_params.max_ang)
            over_max_dist_ang, under_min_dist_ang = step_modified_curvature_test(set_suggestion, prev_set.points,
                                                self.dist, input_params.min_dist_ang, input_params.max_dist_ang)
            dist_reduced = True
            self.next_dist = self.dist
        if ((over_max_ang or over_max_dist_ang)):
            print('Smaller step than min_sep required with current requirements. Continuing anyway')
        # If curvature is very small
        if (under_min_ang and under_min_dist_ang and not dist_reduced):
            self.next_dist = min(self.dist*2, input_params.max_sep)
        else:
            self.next_dist = self.dist

        # Check whether neighboring points are close enough to each other
        add_point_after = max_dist_test(set_suggestion, input_params.max_sep)

        j = 0

        # Insert points wherever points are too far from each other
        for i in add_point_after:
            set_suggestion.insert(i+j+1, Point._find_ordinary_point(i, prev_set, input_params, self.dist,
                                                                    inbetween=True))
            if (not in_domain(set_suggestion[i+j+1].pos, input_params.dom_bound)):
                raise OutsideOfDomainError('Attempted to place point outside domain. Returning manifold')
            j += 1

        # Removing loops
        set_suggestion, loop_deleted = GeodesicLevelSet._remove_loops(set_suggestion, input_params.min_sep,
                                                                      input_params.max_sep)

        if (2*np.pi*geo_dist/input_params.init_num_points < 1.1*min_sep):
            to_be_deleted = []
        else:
            # Check whether neighboring points are far enough from each other
            to_be_deleted = min_dist_test(set_suggestion, input_params.min_sep, input_params.max_sep)

        # Delete points wherever points are too close to each other
        num_removed = 0
        for i in to_be_deleted:
            set_suggestion.pop(i-num_removed)
            num_removed += 1

        if len(set_suggestion) < 5:
            raise InsufficientPointsError('Insufficient number of points to form proper level set.')

        # Setting indices
        for pnt in set_suggestion:
            pnt.index = index
            index += 1

        # Setting triangulations
        self.triangulations = GeodesicLevelSet.add_triangulations(set_suggestion, add_point_after,
                                                                  to_be_deleted, loop_deleted, prev_set)
        return set_suggestion, index

    @staticmethod
    def add_triangulations(set_suggestion, add_point_after, to_be_deleted, loop_deleted, prev_set):
        """GeodesicLevelSet.add_triangulations(set_suggestion, add_point_after, to_be_deleted,
                                               loop_deleted, prev_set)

        A function which (manually) computes triangulations for a completed
        (and accepted) geodesic level set, intended to generate more
        aesthetically pleasing visualizations further down the line.

        param: set_suggestion  -- An accepted _next_ geodesic level set,
                                  as computed by _generate_set and
                                  analyzed by revise_set
        param: add_point_after -- List of point indices after which
                                  points have been added in order to
                                  conform with max_sep constraints
        param: to_be_deleted   -- List of point indices to be deleted,
                                  in order to conform with min_sep
                                  constraints
        param: loop_deleted --    List of point indices to be deleted,
                                  having been identified as sources of
                                  (or the result of) numerical,
                                  nonphysical noise
        param: prev_set --        The most recently computed (and
                                  accepted) geodesic level set

        return: tris -- A list specifying the corners of the
                        triangles which constitute a
                        triangulation of set_suggestion

        """
        links = [Link() for i in range(len(prev_set.points))]

        props = [i for i in range((len(prev_set.points)))]

        j = 0
        for i in add_point_after:
            props.insert(i+j+1,-1)
            j += 1

        loop_del_anc = []
        for i in loop_deleted:
            if (props[i] >= 0):
                links[props[i]].next = next_ind(loop_deleted,i,len(props))
                if (props[last_ind(loop_deleted,i,len(props))] == -1):
                    links[props[i]].last = last_ind(loop_deleted,i,len(props))
                loop_del_anc.append(props[i])

        num_removed = 0
        for i in loop_deleted:
            props.pop(i-num_removed)
            num_removed += 1

        for i in to_be_deleted:
            if (props[i] >= 0):
                links[props[i]].next = next_ind(to_be_deleted,i,len(props))
                if (props[last_ind(to_be_deleted,i,len(props))] == -1):
                    links[props[i]].last = last_ind(to_be_deleted,i,len(props))

        for i in loop_del_anc:
            if (links[i].next in to_be_deleted):
                links[i].next = next_ind(to_be_deleted,links[i].next,len(props))
            else:
                links[i].next -= len(np.where(np.array(to_be_deleted) < links[i].next)[0])
            if (not links[i].last is None):
                if (links[i].last in to_be_deleted):
                    links[i].last = last_ind(to_be_deleted,links[i].last,len(props))
                else:
                    links[i].last -= len(np.where(np.array(to_be_deleted) < links[i].last)[0])

        num_removed = 0
        for i in to_be_deleted:
            props.pop(i-num_removed)
            num_removed += 1

        for i in range(len(props)):
            if (props[i] >= 0):
                links[props[i]].heir = i
                links[props[i]].next = divmod(i+1,len(props))[1]
                if (props[divmod(i-1,len(props))[1]] == -1):
                    links[props[i]].last = divmod(i-1,len(props))[1]

        # Setting up triangles
        tris = []
        for i in range(len(links)):
            if (links[i].heir is None):
                if (links[i].last is None):
                    tris.append([prev_set.points[i].index, prev_set.points[divmod(i+1,
                                len(prev_set.points))[1]].index, set_suggestion[links[i].next].index])
                else:
                    tris.append([prev_set.points[i].index, set_suggestion[links[i].last].index,
                                 set_suggestion[links[i].next].index])
                    tris.append([prev_set.points[i].index, prev_set.points[divmod(i+1,
                                len(prev_set.points))[1]].index, set_suggestion[links[i].next].index])
            else:
                if (links[i].last is None):
                    tris.append([prev_set.points[i].index, prev_set.points[divmod(i+1,len(prev_set.points))[1]].index,
                                set_suggestion[links[i].next].index])
                    tris.append([prev_set.points[i].index, set_suggestion[links[i].heir].index,
                                set_suggestion[links[i].next].index])
                else:
                    tris.append([prev_set.points[i].index, set_suggestion[links[i].last].index,
                               set_suggestion[links[i].heir].index])
                    tris.append([prev_set.points[i].index, set_suggestion[links[i].heir].index,
                               set_suggestion[links[i].next].index])
                    tris.append([prev_set.points[i].index, set_suggestion[links[i].next].index,
                                prev_set.points[divmod(i+1,len(prev_set.points))[1]].index])
        return tris


    @staticmethod
    def _remove_loops(set_suggestion, min_sep, max_sep):
        """GeodesicLevelSet._remove_loops(set_suggestion, min_sep, max_sep)

        A function which detects and removes nonphysical loops in a suggested
        geodesic level set, facilitating extended growth of a manifold.

        *** This function is called by the constructor, explicitly calling this
            function should never be necessary and is not advised in general ***

        param: set_suggestion -- A suggestion for the _next_ geodesic level set,
                                 as computed by _generate_set
        param: min_sep --        The minimum allowed distance separating points
                                 in a geodesic level set
        param: max_sep --        The maximum allowed distance separating points
                                 in a geodesic level set

        return: new_set_suggestion -- A new set suggestion, based upon the input set_suggestion,
                                      where the aforementioned loops have been removed
        return: loop_deleted --       A list of point indices, referring to points which have
                                      been deleted in order to remove nonphysical loops
                                      which arise as a consequence of numerical noise
        """

        did_something = False

        n = len(set_suggestion)
        loop_deleted = []

        if (n > 20):
            to_be_added = np.ones(n,dtype=np.bool)
            seps = np.empty(n)
            for i in range(n):
                seps[i] = cy_norm2(set_suggestion[divmod(i+1,n)[1]].pos - set_suggestion[i].pos)

            for i in range(n):
                # Forward
                for j in range(2,int(n/4)):
                    arcdist = np.sum(seps[i:min(n,i+j)]) + np.sum(seps[0:max(i+j-n,0)])
                    if ((cy_norm2(set_suggestion[divmod(i+j,n)[1]].pos - set_suggestion[i].pos)
                        < min(max_sep, 0.7*arcdist))
                        and (len(np.nonzero(to_be_added[i:min(i+j+1,n)])[0]) == len(to_be_added[i:min(i+j+1,n)]))
                        and (len(np.nonzero(to_be_added[0:max(i+j-n+1,0)])[0]) ==
                             len(to_be_added[0:max(i+j-n+1,0)]))):

                        did_something = True

                        to_be_added[i+1:min(i+j,n)] = False
                        to_be_added[0:max(i+j-n,0)] = False
                # Backward
                for j in range(2,int(n/4)):
                    arcdist = np.sum(seps[max(0,i-j):i]) + np.sum(seps[min(i-j+n,n):n])
                    if ((cy_norm2(set_suggestion[divmod(i-j,n)[1]].pos - set_suggestion[i].pos)
                        < min(max_sep, 0.7*arcdist))
                        and (len(np.nonzero(to_be_added[min(i-j+n,n):n])[0]) ==
                                 len(to_be_added[min(i-j+n,n):n]))
                        and (len(np.nonzero(to_be_added[max(0,i-j):i+1])[0]) ==
                             len(to_be_added[max(0,i-j):i+1]))):

                        did_something = True

                        to_be_added[min(i-j+n+1,n):n] = False
                        to_be_added[max(0,i-j+1):i] = False
                new_set_suggestion = []
                for k in range(n):
                    if (to_be_added[k]):
                        new_set_suggestion.append(set_suggestion[k])
                    else:
                        loop_deleted.append(k)

        else:
            new_set_suggestion = set_suggestion

        if (did_something):
            print('remove_loops did something!')

        loop_deleted = list(set(loop_deleted))
        loop_deleted.sort()

        return new_set_suggestion, loop_deleted

###################### Auxiliary functions for the GeodesicLevelSet class ####################

# Tests whether any steps changed too much in terms of angle from the last steps
def curvature_test(curr_set_points, prev_set_points, min_ang, max_ang):
    over_max_ang, under_min_ang = False, True
    for i in range(len(prev_set_points)):
        dot_prod = np.dot(prev_set_points[i].prev_vec, curr_set_points[i].prev_vec)
        if (np.arccos(np.sign(dot_prod)*min(abs(dot_prod),1)) > min_ang):
            under_min_ang = False
            break
    for i in range(len(prev_set_points)):
        dot_prod = np.dot(prev_set_points[i].prev_vec, curr_set_points[i].prev_vec)
        if (np.arccos(np.sign(dot_prod)*min(abs(dot_prod),1)) > max_ang):
            over_max_ang = True
            break

    return over_max_ang, under_min_ang

# Similar to above, only including step length
def step_modified_curvature_test(curr_set_points, prev_set_points, curr_dist, min_dist_ang, max_dist_ang):
    over_max_dist_ang, under_min_dist_ang = False, True
    for i in range(len(prev_set_points)):
        dot_prod = np.dot(prev_set_points[i].prev_vec, curr_set_points[i].prev_vec)
        if (curr_dist*np.arccos(np.sign(dot_prod)*min(abs(dot_prod),1)) > min_dist_ang):
            under_min_dist_ang = False
            break
    for i in range(len(prev_set_points)):
        dot_prod = np.dot(prev_set_points[i].prev_vec, curr_set_points[i].prev_vec)
        if (curr_dist*np.arccos(np.sign(dot_prod)*min(abs(dot_prod),1)) > max_dist_ang):
            over_max_dist_ang = True
            break
    return over_max_dist_ang, under_min_dist_ang

# Check whether any points on the new level set are too close to each other -> Indicate points to delete
def min_dist_test(curr_set_points, min_sep, max_sep):
    to_be_deleted = []
    n = len(curr_set_points)
    interpoint_dist = np.empty(n)
    for i in range(0,n):
        interpoint_dist[i] = cy_norm2(curr_set_points[divmod(i+1,n)[1]].pos - curr_set_points[i].pos)
        #interpoint_dist[i] = np.linalg.norm(curr_set_points[divmod(i+1,n)[1]].pos - curr_set_points[i].pos)
    i, j = 0, 0

    while (i < n):
        if (interpoint_dist[i] < min_sep and interpoint_dist[i] + min(interpoint_dist[divmod(i-1,n)[1]],
                                                         interpoint_dist[divmod(i+1,n)[1]]) < max_sep):
            if (interpoint_dist[divmod(i-1,n)[1]] < interpoint_dist[divmod(i+1,n)[1]]):
                interpoint_dist[divmod(i-1,n)[1]] += interpoint_dist[i]
                interpoint_dist = np.delete(interpoint_dist,i,0)
                to_be_deleted.append(i + j)
                j += 1
                n -= 1
            else:
                interpoint_dist[i] += interpoint_dist[divmod(i+1,n)[1]]
                interpoint_dist = np.delete(interpoint_dist,divmod(i+1,n)[1],0)
                to_be_deleted.append(i + j)
                j += 1
                n -= 1
        else:
            i += 1

    to_be_deleted.sort()

    return to_be_deleted

# Check whether any points on the new level set are too far from each other -> Indicate where to insert new points
def max_dist_test(curr_set_points, max_sep):
    add_point_after = []
    n = len(curr_set_points)
    interpoint_dist = np.empty(n)
    for i in range(0,n):
        interpoint_dist[i] = cy_norm2(curr_set_points[divmod(i+1,n)[1]].pos - curr_set_points[i].pos)
        #interpoint_dist[i] = np.linalg.norm(curr_set_points[divmod(i+1,n)[1]].pos - curr_set_points[i].pos)
    for i in range(0,n):
        if (interpoint_dist[i] > max_sep):
            add_point_after.append(i)
    return add_point_after

# Check whether a point is in the domain of interest
def in_domain(pos, dom_bound, dom_tol = 0.1):
    xran = dom_bound[1] - dom_bound[0]
    yran = dom_bound[3] - dom_bound[2]
    zran = dom_bound[5] - dom_bound[4]
    return (pos[0] >= dom_bound[0]-dom_tol*xran and pos[0] <= dom_bound[1]+dom_tol*xran and
            pos[1] >= dom_bound[2]-dom_tol*yran and pos[1] <= dom_bound[3]+dom_tol*yran and
            pos[2] >= dom_bound[4]-dom_tol*zran and pos[2] <= dom_bound[5]+dom_tol*zran
           )

# Finding next valid index when heir does not exist (for use in triangulation)
def next_ind(to_be_deleted,i,n):
    for j in range(1,n):
        if (divmod(i+j,n)[1] not in to_be_deleted):
            return divmod(i+j,n)[1] - len(np.where(np.array(to_be_deleted) < divmod(i+j,n)[1])[0])
    return 0

# Finding last valid index when heir does not exist (for use in triangulation)
def last_ind(to_be_deleted,i,n):
    for j in range(1,n):
        if (divmod(i-j,n)[1] not in to_be_deleted):
            return divmod(i-j,n)[1] - len(np.where(np.array(to_be_deleted) < divmod(i-j,n)[1])[0])
    return 0


class Point:
    """A class of which a collection of instances parametrizes a
    geodesic level set.

    Methods defined here:

    Point.__init__(level_num, dist, input_params, prev_set)
    Point._find_ordinary_point(index, prev_set, input_params, dist, inbetween)
    Point._check_ab()
    Point._weighted_prev_vec(index, prev_set, s_lower, s_prev, s_upper)
    Point._weighted_tan_vec(index, prev_set, s_lower, s_prev, s_upper)

    """
    def __init__(self, pos, prev_vec = None, tan_vec = None, index = None):
        """Point.__init__(pos, prev_vec, tan_vec)

        Constructor for a Point object.

        *** This function is called by various other methods
            (classmethods or otherwise) of the Point class ***

        param: pos --      (NumPy) array specifying the (Cartesian) point coordinates
        param: prev_vec -- Normalized vector, as a (NumPy) array, specifying the direction
                           of the straight line from the previous point to this one.
        param: tan_vec --  Normalized vector, as a (NumPy) array, specifying the local
                           tangential vector, used in order to define a half-plane
                           'radially' outwards from _this_ point.

        """
        # Remember my position
        self.pos = pos
        # Remember my "previous" vector
        self.prev_vec = prev_vec
        # Remember my "tangential" vector
        self.tan_vec = tan_vec
        # Remember my index
        self.index = index

        # The following member variables are not set upon construction,
        # as they are not needed prior to the LCS candidate selection
        # process. Nevertheless, 'allocating' them decreases the
        # indexing speed later, alas:
        # Remember if I satisfy conditions A, B, and D
        self.in_ab = None
        # Remember my lambda3 value
        self.lambda3 = None
        # Remember my "weight" (~ surrounding area)
        self.weight = None

    @classmethod
    def _find_ordinary_point(cls, index, prev_set, input_params, dist, inbetween):
        if inbetween:
            s_lower = prev_set.interpolation.s[index]
            s_upper = prev_set.interpolation.s[divmod(index+1,len(prev_set.interpolation.s))[1]]
            ds = min(abs(s_upper - s_lower), abs(s_upper-s_lower + 1))
            s_prev = divmod(s_lower + 0.5*ds, 1)[1]

            prev_vec = Point._weighted_prev_vec(index, prev_set, s_lower, s_prev, s_upper)
            tan_vec = Point._weighted_tan_vec(index, prev_set, s_lower, s_prev, s_upper)
            prev_pos = prev_set.interpolation(s_prev)

        else:
            tan_vec = prev_set.points[index].tan_vec
            prev_pos = prev_set.points[index].pos
            prev_vec = prev_set.points[index].prev_vec

        max_arclen = input_params.max_arclen_factor*dist

        arclen = 0
        pos_curr = prev_pos

        max_stride = dist
        stride = 0.1*max_stride

        direction_generator.set_tan_vec(tan_vec)
        direction_generator.set_prev_vec(prev_vec)

        strain_integrator.set_aim_assister(direction_generator)

        while(arclen < max_arclen and abs(cy_norm2(pos_curr-prev_pos) - dist) > dist*input_params.dist_tol):
            stride = np.minimum(stride, max_stride)
            nu_prv_vec = direction_generator(arclen,pos_curr)
            arclen, pos_curr, stride = strain_integrator(arclen, pos_curr, stride)
            direction_generator.set_prev_vec(nu_prv_vec)
            max_stride = cy_max(dist*input_params.dist_tol, (dist-cy_norm2(pos_curr-prev_pos)))
            #max_stride = cy_min(cy_max(dist,cy_norm2(pos_curr-prev_pos)),dist*input_params.dist_tol)

        if arclen >= max_arclen:
            raise NeedSmallerDistError('Need smaller dist')

        return cls(pos_curr, cy_normalize(pos_curr - prev_pos), tan_vec)

    def _is_in_ab(self):
        """Point._is_in_ab()

        Checks whether or not the point satisfies the A and B criteria
        for strong LCSs.

        The boolean flag 'in_ab' is set to True or False, accordingly.
        """

        A = lm3_field(self.pos) > lm2_field(self.pos) and lm3_field(self.pos) > 1
        B = np.linalg.multi_dot((xi3_field(self.pos),lm3_field.hess(self.pos),xi3_field(self.pos))) <= 0

        self.in_ab = A and B

##################################################################################################
##################################### Helping functions ##########################################
##################################################################################################

    # Computes weighted average of "from previous" vectors at neighboring points
    @staticmethod
    def _weighted_prev_vec(index, prev_set, s_lower, s_prev, s_upper):
        """Point._weighted_prev_vec(index, prev_set, u_lower, s_prev, s_upper)

        Computes a weighted average of 'from previous' vectors at neighboring
        points. Relevant when adding points inbetween others in order to
        conform with demands regarding minimum and maximum separation.

        param: index --    Index of the point in the previous set (prev_set),
                           after which a new fictitious point is to be generated
                           inbetween index and index+1 in order to compute
                           local radial vectors.
        param: prev_set -- The most recently computed (and accepted)
                           geodesic level set
        param: s_lower --  The lower permitted limit for the s parameter
        param: s_prev --   The s parameter of the parametrization of the
                           previous set, which in the context of the
                           B-spline parametrization of prev_set
                           yields the Cartesian coordinates of the
                           coordinates of prev_set.points[index]
        param: s_upper --  The upper permitted limit for the s parameter

        return: wt_vec -- Normalized vector (as a NumPy array) from weighted
                          average of 'from previous' vectors at neighboring
                          points (namely the ones located at index and index+1)

        """
        ds_upper = min(abs(s_upper - s_prev), abs(s_upper - s_prev + 1))
        ds_lower = min(abs(s_prev - s_lower), abs(s_prev - s_lower + 1))
        ds_upper = ds_upper/(ds_upper + ds_lower)
        ds_lower = 1 - ds_upper
        return cy_normalize(ds_lower*prev_set.points[divmod(index+1,len(prev_set.points))[1]].prev_vec \
                + ds_upper*prev_set.points[index].prev_vec)

    # Computes weighted average of tangential vectors at neighboring points
    @staticmethod
    def _weighted_tan_vec(index, prev_set, s_lower, s_prev, s_upper):
        """Point._weighted_tan_vec(index, prev_set, u_lower, s_prev, s_upper)

        Computes a weighted average of (approximately) tangential vectors at
        neighboring points. Relevant when adding points inbetween others in
        order to conform with demands regarding minimum and maximum separation.

        param: index --    Index of the point in the previous set (prev_set),
                           after which a new fictitious point is to be generated
                           inbetween index and index+1 in order to compute
                           local radial vectors.
        param: prev_set -- The most recently computed (and accepted)
                           geodesic level set
        param: s_lower --  The lower permitted limit for the s parameter
        param: s_prev --   The s parameter of the parametrization of the
                           previous set, which in the context of the
                           B-spline parametrization of prev_set
                           yields the Cartesian coordinates of the
                           coordinates of prev_set.points[index]
        param: s_upper --  The upper permitted limit for the s parameter

        return: wt_vec -- Normalized vector (as a NumPy array) of the (nearly)
                          tangential vectors at neighboring points
                          (namely the ones located at index and index+1)

        """
        ds_upper = min(abs(s_upper - s_prev), abs(s_upper - s_prev + 1))
        ds_lower = min(abs(s_prev - s_lower), abs(s_prev - s_lower + 1))
        ds_upper = ds_upper/(ds_upper + ds_lower)
        ds_lower = 1 - ds_upper
        return cy_normalize(ds_lower*prev_set.points[divmod(index+1,len(prev_set.points))[1]].tan_vec \
                + ds_upper*prev_set.points[index].tan_vec)

##### Error classes #####

class PointNotFoundError(Exception):
    def __init__(self, value):
        self.value = value

class NeedSmallerDistError(Exception):
    def __init__(self, value):
        self.value = value

class OutsideOfDomainError(Exception):
    def __init__(self, value):
        self.value = value

class CannotDecreaseDistFurtherError(Exception):
    def __init__(self, value):
        self.value = value

class InsufficientPointsError(Exception):
    def __init__(self, value):
        self.value = value

##### Other classes #####

class Link:
    def __init__(self):
        self.heir = None
        self.next = None
        self.last = None


class LCSCandidate:
    def __init__(self, manifold):
        max_dist_multiplier = 2

        self.avg_lambda3 = 0
        self.mf_orig_geo_dist = manifold.geo_dist
        self.points = [Point(manifold.input_params.init_pos)]
        self.indices = [0]
        self.points[0].in_ab = True
        self.points[0].lambda3 = lm3_field(self.points[0].pos)
        if not hasattr(manifold,'levelsets'):
            return
        else:
            if not hasattr(manifold.levelsets[0],'points'):
                return
            else:

                init_radius = cy_norm2(manifold.levelsets[0].points[0].pos-self.points[0].pos)
                self.points[0].weight = np.pi*(0.5*init_radius)**2


                pts_in_ab = []
                pts_not_in_ab = []

                for l in manifold.levelsets:
                    for p in l.points:
                        if p.in_ab:
                            pts_in_ab.append(p)
                        else:
                            pts_not_in_ab.append(p)

                dist_thresh = max_dist_multiplier*manifold.input_params.max_sep

                appended_coords = np.array(self.points[0].pos)

                for p in pts_in_ab:
                    if len(appended_coords.shape) == 1:
                        if (np.linalg.norm(appended_coords - p.pos) < dist_thresh and
                        in_domain(p.pos, manifold.input_params.dom_bound, dom_tol = 0)):
                            self.points.append(p)
                            self.indices.append(p.index)
                            appended_coords = np.vstack((appended_coords,p.pos))
                    else:
                        if (np.any(np.less(np.linalg.norm(appended_coords-p.pos,axis=1),dist_thresh)) and
                        in_domain(p.pos, manifold.input_params.dom_bound, dom_tol = 0)):
                            self.points.append(p)
                            self.indices.append(p.index)
                            appended_coords = np.vstack((appended_coords,p.pos))

                if len(appended_coords.shape) == 1:
                    for p in pts_not_in_ab:
                        if (np.any(np.less(np.linalg.norm(p.pos - appended_coords), dist_thresh)) and
                        in_domain(p.pos, manifold.input_params.dom_bound, dom_tol = 0)):
                            self.points.append(p)
                            self.indices.append(p.index)
                else:
                    for p in pts_not_in_ab:
                        if (np.any(np.less(np.linalg.norm(p.pos - appended_coords, axis=1), dist_thresh)) and
                        in_domain(p.pos, manifold.input_params.dom_bound, dom_tol = 0)):
                            self.points.append(p)
                            self.indices.append(p.index)

                self.tot_weight = 0
                for point in self.points:
                    self.avg_lambda3 += point.lambda3*point.weight
                    self.tot_weight += point.weight

                self.avg_lambda3 /= self.tot_weight
                self.triangulations = []

                for tri in manifold.triangulations:
                    if (set(tri).issubset(set(self.indices))):
                        self.triangulations.append(tri)

                for tri in self.triangulations:
                    for i in range(3):
                        tri[i] = self.indices.index(tri[i])

                self.xs = [pts.pos[0] for pts in self.points]
                self.ys = [pts.pos[1] for pts in self.points]
                self.zs = [pts.pos[2] for pts in self.points]


######################### Functions for selection LCSs from LCS candidates ###########################

def identify_lcs(lcs_candidates, xs, ys, zs):

    lcs_candidates = np.array(lcs_candidates)
    lcs_candidates = lcs_candidates[[len(lcs_candidate.points) > 100 for lcs_candidate in lcs_candidates]]
    lcd_candidates = list(lcs_candidates)
    lcs_candidates = sort_lcs_candidates(lcs_candidates)
    lcss = []
    presences = np.zeros((len(lcs_candidates),len(xs),len(ys),len(zs)),dtype=np.bool)

    for i, lcs_candidate in enumerate(lcs_candidates):
        for point in lcs_candidate.points:
            idx, idy, idz = region_id(point.pos, xs, ys, zs)
            presences[i,idx,idy,idz] = True

    for l in range(len(zs)):
        for k in range(len(ys)):
            for j in range(len(xs)):
                for i, lcs_candidate in enumerate(lcs_candidates):
                    # LCS_candidates sorted by max lambda3 -> First encountered is the LCS
                    if presences[i,j,k,l]:
                        lcss.append(lcs_candidate)
                        break

    return list(set(lcss))


def region_id(pos, xs, ys, zs):
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    dz = zs[1] - zs[0]

    idx = int(divmod(pos[0],dx)[0])
    idy = int(divmod(pos[1],dy)[0])
    idz = int(divmod(pos[2],dz)[0])

    return idx, idy, idz

def sort_lcs_candidates(lcs_candidates):
    lambda3s = [lcs_candidate.avg_lambda3 for lcs_candidate in lcs_candidates]
    indices = np.argsort(lambda3s)[::-1]
    return [lcs_candidates[i] for i in indices]

###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################
###################################################################################

def parallelized_lcs_selection(start_ind,end_ind,q):
    mfs_mn = []
    mfs_up = []
    mfs_dn = []
    ls = []
    for i in range(start_ind,end_ind):
        mfs_mn.append(np.load('manifolds/{0:04d}_manifold_m.npy'.format(i))[0])
        mfs_up.append(np.load('manifolds/{0:04d}_manifold_u.npy'.format(i))[0])
        mfs_dn.append(np.load('manifolds/{0:04d}_manifold_d.npy'.format(i))[0])

    for outer in zip(mfs_mn, mfs_up, mfs_dn):
        for mf in outer:
            mf.check_ab()
            mf.compute_lambda3_and_weights()

    LCS_m = [LCSCandidate(mf) for mf in mfs_mn]
    LCS_u = [LCSCandidate(mf) for mf in mfs_up]
    LCS_d = [LCSCandidate(mf) for mf in mfs_dn]

   #LCSs_init = [LCSCandidate(mf) for mf in mfs]

    for (u,m,d) in zip(LCS_u, LCS_m, LCS_d):
        if(m.avg_lambda3 > max(u.avg_lambda3,d.avg_lambda3)
            and m.tot_weight > 1):
#and m.mf_orig_geo_dist > 0.5 and len(m.points) > 100):
            ls.append(m)

    q.put(ls)

if __name__ == '__main__':

   num_files = len(next(os.walk('manifolds'))[2])


   x, y, z = np.load('precomputed_strain_params/x.npy'), np.load('precomputed_strain_params/y.npy'), np.load('precomputed_strain_params/z.npy')

   lm2 = np.load('precomputed_strain_params/lm2.npy')
   lm3 = np.load('precomputed_strain_params/lm3.npy')

   xi3 = np.load('precomputed_strain_params/xi3.npy')

   lm2_field = TrivariateSpline(x,y,z,lm2,4,4,4)
   lm3_field = TrivariateSpline(x,y,z,lm3,4,4,4)

   xi3_field = LinearEigenvectorInterpolator(x,y,z,xi3)
   print('Evidence')

   LCSs = []

   nproc = 16

   div = np.ceil(np.linspace(0,int(num_files/3),nproc+1)).astype(int)
   qs = [mp.Queue() for j in range(nproc)]
   ps = [mp.Process(target = parallelized_lcs_selection,
                   args = (div[j],div[j+1],qs[j])) for j in range(nproc)]
   for p in ps:
       p.start()
   for q in qs:
       foo = q.get()
       if len(foo):
           for f in foo:
               LCSs.append(f)
   for p in ps:
       p.join()

   path = 'LCSs'

   ensure_path_exists(path)

   np.save(path+'/lcss.npy',LCSs)


