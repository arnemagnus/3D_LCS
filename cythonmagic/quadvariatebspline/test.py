from quadvariatevectorinterpolation import ScalarQuadvariateSpline, TrivectorQuadvariateSpline

import numpy as np

t = np.linspace(0,10,15)

x = np.linspace(0,2,11)

y = np.linspace(0,2,12)

z = np.linspace(0,2,13)
f = np.empty((3,t.shape[0]*x.shape[0]*y.shape[0]*z.shape[0]))

f[0] = np.multiply.outer(np.cos(t), np.sin(np.pi*np.meshgrid(x,y,z,indexing='ij')[0])).ravel(order='F')
f[1] = np.multiply.outer(np.cos(t), np.sin(np.pi*np.meshgrid(x,y,z,indexing='ij')[1])).ravel(order='F')
f[2] = np.multiply.outer(np.cos(t), np.sin(np.pi*np.meshgrid(x,y,z,indexing='ij')[2])).ravel(order='F')

a = TrivectorQuadvariateSpline(t,x,y,z,f)

print(a(0,np.array((0.5,0.5,0.5))))
print(a.jac(0,np.array((0.5,0.5,0.5))))
