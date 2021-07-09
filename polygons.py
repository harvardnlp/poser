from numba import jit, njit
import numba
import numpy as np 

@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside


@njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean) 
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D

import torch

def perp(a) :
    b = torch.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a, b):
    a = a.view(-1, 1, 2)
    b = b.view(1, -1, 2)
    da = a[:, :, 1]-a[:, :, 0]
    db = b[:, :, 1]-b[:, :, 0]
    dp = a[:, :, 0]-b[:, :, 0]
    dap = perp(da)
    denom = torch.einsum("abd,abd-> ab", dap, db)
    num = torch.einsum("abd,abd-> ab", dap, dp)
    return (num / denom.float()) + b[:, :, 0]

p1 = array( [0.0, 0.0] )
p2 = array( [1.0, 0.0] )

p3 = array( [4.0, -5.0] )
p4 = array( [4.0, 2.0] )

print seg_intersect( p1,p2, p3,p4)
