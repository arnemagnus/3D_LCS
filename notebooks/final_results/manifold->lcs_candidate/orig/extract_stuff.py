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

mf = np.load('mf.npy')[0]


np.save('mf_xs.npy', mf.xs)
np.save('mf_ys.npy', mf.ys)
np.save('mf_zs.npy', mf.zs)
np.save('mf_tris.npy', mf.triangulations)

lcs = np.load('lcs.npy')[0]


np.save('lcs_xs.npy', lcs.xs)
np.save('lcs_ys.npy', lcs.ys)
np.save('lcs_zs.npy', lcs.zs)
np.save('lcs_tris.npy', lcs.triangulations)
