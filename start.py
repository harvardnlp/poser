
# ICFP Contest
import pandas as pd
import skgeom as sg
from skgeom.draw import draw
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
__st.set_option('deprecation.showPyplotGlobalUse', False)

poses = json.load(open("p1.json"))

epsilon = poses["epsilon"]
figure = poses["figure"]
graph = np.array(figure["edges"])
vertices = [sg.Point2(a[0], a[1]) for a in figure["vertices"]]
edge_segments = [sg.Segment2(vertices[fro], vertices[to])
                 for (fro, to) in graph]
poly = sg.Polygon([sg.Point2(p[0], p[1]) for p in poses["hole"]])

if False:
    fig = plt.Figure()
    draw(poly)
    for segment in edge_segments:
        draw(segment)
    __st.pyplot()



holepts = torch.tensor(poses["hole"], dtype=float)
original = torch.tensor(figure["vertices"], dtype=float)
npts = original.shape[0]
graph = torch.tensor(graph)
parameters = original.clone()
parameters.requires_grad_(True)
def seg_dist(p):
    constraint = p[graph[:, 0]] - p[graph[:, 1]]
    return (constraint * constraint).sum(-1)

target = epsilon / 1000000

shape = original.shape
shape

# The main constraint

def stretch_constraint(p):
    return torch.relu(((seg_dist(p) / seg_dist(original)) - 1.0).abs() - target)

def spring_constraint(p):
    d = (seg_dist(p) - seg_dist(original))
    return d.abs()



def dist(u, v):
    d = u - v
    return (d * d).sum(-1)

x = torch.clamp(torch.tensor([1,2,3]), max=1)
x

def euclidean_projection(p, v, w):
    p = p.view(-1, 1, 2)
    v = v.view(-1, 2)
    w = w.view(-1, 2)
    l2 = dist(v, w)
    t = torch.clamp(torch.einsum("psd,sd->ps", p - v, w - v) / l2, max=1, min=0)
    projection = v + t[..., None] * (w - v)
    return dist(p, projection).sqrt()

x = euclidean_projection(torch.tensor([[1, 0], [3, 0.0], [4, 0.0]]),
                         torch.tensor([[0, -1.0] , [1, -1.0]]),
                         torch.tensor([[0, 1.0], [1, 1.0]]))
x

def outside_constraint(points):
    outer = [i for i, p in enumerate(points)
             if poly.oriented_side(sg.Point2(p[0], p[1])) == sg.Sign.NEGATIVE]
    p = outer_points = points[outer]
    
    s = holepts.shape[0]
    v = start_hole = holepts[:]
    w = end_hole = holepts[list(range(1, s)) + [0]]
    return euclidean_projection(p, v, w).min(-1)[0], p, outer

def inside_constraint(points):
    inside = [i for i, p in enumerate(points)
             if poly.oriented_side(sg.Point2(p[0], p[1])) == sg.Sign.POSITIVE]
    p = inner_points = points[inside]
    
    s = holepts.shape[0]
    v = start_hole = holepts[:]
    w = end_hole = holepts[list(range(1, s)) + [0]]
    return euclidean_projection(p, v, w).min(-1)[0], p, inside


    # // Consider the line extending the segment, parameterized as v + t (w - v).
    # // We find projection of point p onto the line. 
    # // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    # // We clamp t from [0,1] to handle points outside the segment vw.
    
x = poly.oriented_side(sg.Point2(60, 60)) == sg.Sign.NEGATIVE
x
    
# The objective

def objective(p):
    constraint = p.view(npts, 1, 2) - holepts.view(1, -1, 2)
    constraint = constraint * constraint
    return constraint.min(-1)[0].sum()

lambd = torch.ones([1], requires_grad=True)
rate = 0.2
opt = torch.optim.SGD([parameters], lr=rate)
if True:
    for epochs in range(10000):
        if (epochs % 1000) == 0 or epochs == 999:
            plt.clf()
            draw(poly)
            vertices = [sg.Point2(a[0], a[1]) for a in parameters.detach().numpy()]
            edge_segments = [sg.Segment2(vertices[fro], vertices[to])
                             for (fro, to) in graph]
            for segment in edge_segments:
                draw(segment)
            __st.pyplot()

        opt.zero_grad()
        spring_cons = spring_constraint(parameters).mean()
        st_cons = stretch_constraint(parameters).sum()
        outside_cons, violators, v_index = outside_constraint(parameters)
        inside_cons, _, _ = inside_constraint(parameters)
        int_cons = (parameters - parameters.round()).abs().mean()
        # loss = objective(parameters) + st_cons + outside_cons

        if epochs < 8000:
            loss = -0.02 * inside_cons.sum() + 0.1 * outside_cons.sum() +  spring_cons  + st_cons 
        else:
            loss =  0.1 * outside_cons.sum() + st_cons + 0.01 * spring_cons
        loss.backward()
        delta = (rate * parameters.grad)
        parameters.data -= delta
        if epochs > 8000:
            decay = 0.01
            roundies = (parameters.data - parameters.data.round()).abs() <= decay
            parameters.data[roundies] =  parameters.data[roundies].round()
            noroundies = (parameters.data - parameters.data.round()).abs() > decay
            parameters.data[noroundies] -=  decay * torch.sign(parameters.data[noroundies] - parameters.data[noroundies].round())
        # opt.step()
        # for ptindex in range(parameters.shape[0]):
        #     for rate in [2.0, 1, 0.5, 0.1]:
        #         delta = (rate * parameters.grad[ptindex]).round()
        #         parameters.data[ptindex] -= delta
        #         if stretch_constraint(parameters).sum() == 0.0:
        #             break
        #         parameters.data[ptindex] += delta
            
        if epochs % 100 == 0.0:
            __st.write("Epoch", epochs)
            p = parameters.round()        
            d = {"loss": loss.detach().item(),
                 "spring" : spring_cons.detach().item(),
                 "stretch" : st_cons.detach().item(),
                 "outside" : outside_cons.sum().detach().item(),
                 "int cons" : int_cons.item(),
                 "round stretch": stretch_constraint(p).sum().item(),
                 "round outside": outside_constraint(p)[0].sum().item()
                 # "int con" : int_cons.detach().item()
            }
            
            __st.write(d)

            # print(p)
            # print(stretch_constraint(p).sum())

            
            # __st.write("violators", violators, outside_cons)
            # # for v in v_index:
            # #     __st.write(parameters[v], parameters.grad[v])
            # __st.write("stretch violation", st_cons)
            # __st.write("outside violation", outside_cons)
        # __st.write("violation cross", st_cons)

        # for v in v_index:
        #     __st.write(parameters[v], parameters.grad[v])

p = parameters.round()
        
print(p)
print(stretch_constraint(p).sum())
  
