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
        if y >= min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x < xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside
from matplotlib.path import Path
p = Path([[0, 0], [1, 0], [1, 1], [0, 1.0]])
grid = p.contains_points(np.array([[0.9, -0.01]]), radius=0.0001)
print(grid)
# print(pointinpolygon(0.9, 0.0, np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])))
# print(pointinpolygon(0.9, 1.0, np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])))
# print(pointinpolygon(1.0, 0.0, np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])))
# print(pointinpolygon(0.0, 0.0, np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])))

# @njit(parallel=True)
def parallelpointinpolygon(points, polygon):
    p = Path(polygon)
    return p.contains_points(points, radius=0.001)

    # D = np.empty(len(points), dtype=numba.boolean) 
    # for i in numba.prange(0, len(D)):
    #     D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    # return D

import torch

# def perp(a) :
#     b = torch.empty_like(a)
#     b[...,0] = -a[...,1]
#     b[...,1] = a[...,0]
#     return b

# # line segment a given by endpoints a1, a2
# # line segment b given by endpoints b1, b2
# # return 
# def seg_intersect(a1, a2, b1, b2):
#     a1 = a1.view(-1, 1, 2)
#     a2 = a2.view(-1, 1, 2)
#     b1 = b1.view(1, -1, 2)
#     b2 = b2.view(1, -1, 2)
    
#     da = a2 - a1
#     db = b2 - b1
#     dp = a1 - b1
#     dap = perp(da)
#     denom = torch.einsum("abd,abd-> ab", dap, db)
#     num = torch.einsum("abd,abd-> ab", dap, dp)
#     print(num, denom)
#     # return denom != 0.0

#     return (num / denom.float())[..., None] * db + b1

def ccw(A,B,C):
    return (C[...,1] - A[...,1]) * (B[...,0]-A[...,0]) > (B[...,1]-A[...,1]) * (C[...,0]-A[...,0])

# Return true if line segments AB and CD intersect
def seg_intersect(A,B,C,D):
    A = A.view(-1, 1, 2)
    B = B.view(-1, 1, 2)
    C = C.view(1, -1, 2)
    D = D.view(1, -1, 2)
    
    return (ccw(A,C,D) != ccw(B,C,D)) & (ccw(A,B,C) != ccw(A,B,D))

p1 = torch.tensor( [[0.0, -4.0], [1.0, 4.0]] )
p2 = torch.tensor( [[0.0, 5.0], [1.0, 4.0]] )

p3 = torch.tensor( [[-1.0, 1.6], [1.0, 3.0]] )
p4 = torch.tensor( [[2.0, 1.6], [1.0, 5.0]] )

print(seg_intersect( p1,p2, p3,p4))
