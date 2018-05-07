import triangleintersectioncheck
import numpy as np
a = np.array(((1.,0,0),(3,0,0),(2,2,0)))
b = np.array(((1.,0,2),(3,0,2),(2,1,-1)))


A = triangleintersectioncheck.Triangle3D(a)
B = triangleintersectioncheck.Triangle3D(b)

checker = triangleintersectioncheck.MollerTrumboreChecker(eps=1e-8, culling =True)

print(checker(A,B))

print('*'*80)

checker = triangleintersectioncheck.MollerTrumboreChecker(eps=1e-8, culling = False)

print(checker(A,B))

