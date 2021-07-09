
# ICFP Contest

import skgeom as sg
from skgeom.draw import draw
import json
import matplotlib.pyplot as plt
import numpy as np
__st.set_option('deprecation.showPyplotGlobalUse', False)

poses = json.load(open("p1.json"))

poses

epsilon = poses["epsilon"]
figure = poses["figure"]
graph = np.array(figure["edges"])
vertices = [sg.Point2(a[0], -a[1]) for a in figure["vertices"]]
edge_segments = [sg.Segment2(vertices[fro], vertices[to])
                 for (fro, to) in graph]
poly = sg.Polygon([sg.Point2(p[0], -p[1]) for p in poses["hole"]])


fig = plt.Figure()
draw(poly)
for segment in edge_segments:
    draw(segment)
__st.pyplot()

import torch

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

# The objective

def objective(p):
    constraint = p.view(npts, 1, 2) - holepts.view(1, -1, 2)
    constraint = constraint * constraint
    return constraint.min(-1)[0].sum()

lambd = torch.ones([1], requires_grad=True)
opt = torch.optim.SGD([parameters, lambd], lr=0.01)
for epochs in range(10):
    opt.zero_grad()
    st_cons = stretch_constraint(parameters).sum()
    loss = objective(parameters) + lambd[0] * st_cons 
    loss.backward()
    __st.write("loss", loss)
    __st.write("violation", st_cons)
    opt.step()
