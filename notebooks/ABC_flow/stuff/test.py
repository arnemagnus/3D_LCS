import numpy as np

x = np.load('l_x.npy')
y = np.load('l_y.npy')
z = np.load('l_z.npy')
t = np.load('l_t.npy')

from mayavi import mlab
for i in [0,1,4,5,6,7,9,10]:
    s = np.zeros_like(x[i])
    s.fill(i)
    msh = mlab.triangular_mesh(x[i],y[i],z[i],t[i],color=(i/10.,i/10.,i/10.))

mlab.axes(xlabel='x',ylabel='y',zlabel='z',ranges=(0,2*np.pi,0,2*np.pi,0,2*np.pi))
mlab.show()
