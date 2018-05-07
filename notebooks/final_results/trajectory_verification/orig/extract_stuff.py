import numpy as np


class Manifold:
    pass

class InputManifoldParameters:
    pass

class GeodesicLevelSet:
    pass

class Point:
    pass

mf = np.load('mf.npy')[0]


np.save('mf_xs.npy', mf.xs)
np.save('mf_ys.npy', mf.ys)
np.save('mf_zs.npy', mf.zs)
np.save('mf_tris.npy', mf.triangulations)
