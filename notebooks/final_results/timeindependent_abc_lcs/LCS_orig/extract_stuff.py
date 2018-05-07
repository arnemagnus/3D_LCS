import numpy as np

class LCSCandidate:
    pass

class Manifold:
    pass

class InputManifoldParameters:
    pass

class GeodesicLevelSet:
    pass

class Point:
    pass

LCSs = np.load('lcss_max_dist_multplier=1.8_min_weight=6.0.npy')

np.save('xs.npy',[l.xs for l in LCSs])
np.save('ys.npy',[l.ys for l in LCSs])
np.save('zs.npy',[l.zs for l in LCSs])
np.save('tris.npy',[l.triangulations for l in LCSs])
