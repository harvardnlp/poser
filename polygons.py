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
    b[...,0] = -a[...,1]
    b[...,1] = a[...,0]
    return b

# line segment a given by endpoints a1, a2
# line segment b given by endpoints b1, b2
# return 
def seg_intersect(a1, a2, b1, b2):
    a1 = a1.view(-1, 1, 2)
    a2 = a2.view(-1, 1, 2)
    b1 = b1.view(1, -1, 2)
    b2 = b2.view(1, -1, 2)
    
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = perp(da)
    denom = torch.einsum("abd,abd-> ab", dap, db)
    num = torch.einsum("abd,abd-> ab", dap, dp)
    return denom != 0
    # print(num, denom)
    # return (num / denom.float()) + b1


p1 = torch.tensor( [[0.0, 4.0]] )
p2 = torch.tensor( [[1.0, 4.0]] )

p3 = torch.tensor( [[1.5, 5.0]] )
p4 = torch.tensor( [[2.0, 5.0]] )

print(seg_intersect( p1,p2, p3,p4))
