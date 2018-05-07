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

minseps = [0.16,0.08,0.04,0.01]

for m in minseps:
    mf = np.load('mf_minsep={}.npy'.format(m))[0]
    np.save('xs_minsep={}.npy'.format(m),mf.xs)
    np.save('ys_minsep={}.npy'.format(m),mf.ys)
    np.save('zs_minsep={}.npy'.format(m),mf.zs)
    np.save('tris_minsep={}.npy'.format(m),mf.triangulations)

#LCSs = np.load('lcss_max_dist_multplier=1.8_min_weight=6.0.npy')

#np.save('xs.npy',[l.xs for l in LCSs])
#np.save('ys.npy',[l.ys for l in LCSs])
#np.save('zs.npy',[l.zs for l in LCSs])
#np.save('tris.npy',[l.triangulations for l in LCSs])
