# ICFP Contest
import pandas as pd
import skgeom as sg
from skgeom.draw import draw
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
# __st.set_option('deprecation.showPyplotGlobalUse', False)
from polygons import *
import polygons
def dist(u, v):
    d = u - v
    return (d * d).sum(-1)

def euclidean_projection(p, v, w):
    p = p.view(-1, 1, 2)
    v = v.view(-1, 2)
    w = w.view(-1, 2)
    l2 = dist(v, w)
    t = torch.clamp(torch.einsum("psd,sd->ps", p - v, w - v) / l2, max=1, min=0)
    projection = v + t[..., None] * (w - v)
    d = torch.linalg.norm(p-projection)
    return d

class Params:
    def __init__(self, starting_params):
        self.parameters = starting_params.clone()
        self.parameters.requires_grad_(True)
        # x and y bias
        self.bias = torch.zeros(2)
        self.bias.requires_grad_(True)

    def get_parameters(self):
        return self.parameters + self.bias

    def update(self, rate):
        self.parameters.data -= rate * self.parameters.grad
        self.bias.data -= rate * self.bias.grad

        # just absorb back into parameters
        self.parameters.data = self.parameters.data + self.bias.data
        self.bias.data.zero_()

    def set_parameters(self, parameters):
        self.parameters.data.copy_(parameters.data)
        self.bias.data.zero_()

    def raw_parameters(self):
        return [self.parameters, self.bias]

class Problem:
    def __init__(self, problem_number):
        self.problem_number = problem_number
        poses = json.load(open("p%d.json"%problem_number))
        self.epsilon = poses["epsilon"]
        self.figure = poses["figure"]
        self.graph = np.array(self.figure["edges"])
        self.vertices = [sg.Point2(a[0], a[1]) for a in self.figure["vertices"]]
        self.edge_segments = [sg.Segment2(self.vertices[fro], self.vertices[to])
                              for (fro, to) in self.graph]
        self.poly = sg.Polygon([sg.Point2(p[0], p[1]) for p in poses["hole"]])
        self.poly_np = np.array(poses["hole"])
        self.holepts = torch.tensor(poses["hole"], dtype=float)
        self.original = torch.tensor(self.figure["vertices"], dtype=float)
        self.npts = self.original.shape[0]
        self.graph = torch.tensor(self.graph)
        self.target = self.epsilon / 1000000


    def find_intersections(self, p):
        bad_edges = []
        s = self.holepts.shape[0]
        inpoly = parallelpointinpolygon(p.detach().numpy(), self.poly_np)
        # vertices = [sg.Point2(a[0], a[1]) for a in p.detach().numpy()]
        p = p.detach()
        edge_segments = [(i, p[fro], p[to])
                         for i, (fro, to) in enumerate(self.graph)
                         if inpoly[fro] and inpoly[to]]
        # edge_segments = [sg.Segment2(*e) for e in edge_segments]

        a1 = torch.tensor([[v[1][0], v[1][1]] for v in edge_segments])
        a2 = torch.tensor([[v[2][0], v[2][1]] for v in edge_segments])
        b1 = torch.tensor(self.holepts[:])
        b2 = torch.tensor(self.holepts[list(range(1, s)) + [0]])

        n = seg_intersect(a1, a2, b1, b2)

        bad = n.any(dim=1)
        for i in range(bad.shape[0]):
            if bad[i]:
                bad_edges.append(edge_segments[i][0])
        # for i, edge_segment in enumerate(edge_segments):
        #     for edge in self.poly.edges:
        #         if sg.intersection(edge, edge_segment):
        #             bad_edges.append(i)
        #             break
        return bad_edges

    def seg_dist(self, p):
        constraint = p[self.graph[:, 0]] - p[self.graph[:, 1]]
        return (constraint * constraint).sum(-1)

    # The main constraint

    def stretch_constraint(self, p):
        return torch.relu(((self.seg_dist(p) / self.seg_dist(self.original)) - 1.0).abs() - self.target)

    def spring_constraint(self, p):
        d = (self.seg_dist(p) - self.seg_dist(self.original))
        return d.abs()

    def outside_constraint(self, points):
        # outer2 = [i for i, p in enumerate(points)
        #          if self.poly.oriented_side(sg.Point2(p[0], p[1])) == sg.Sign.NEGATIVE]
        inpoly = parallelpointinpolygon(points.detach().numpy(), self.poly_np)
        # outer2 = outer2
        # for i in range(outer.shape[0]):
        #     assert (~outer)[i] == (i in outer2) , "%s"%((points[i], self.poly, outer[i]),)
        p = points[~inpoly]
        s = self.holepts.shape[0]
        v = self.holepts[:]
        w = self.holepts[list(range(1, s)) + [0]]
        d = euclidean_projection(p, v, w).min(-1)[0]
        # p = points[outer2]
        # d2 = euclidean_projection(p, v, w).min(-1)[0]
        # print(d2)
        return d[d!=0.0], p, ~inpoly

    def random_constraint(self, points):
        intersections = self.find_intersections(points)
        new_points = []
        r = torch.rand(100, 1, 1)
        points = r * points[self.graph[intersections, 0]]  + (1-r) * points[self.graph[intersections,1]]

        outer = []
        for i in range(points.shape[1]):
            outer_for_i = []
            inpoly = parallelpointinpolygon(points[:, i].detach().numpy(), self.poly_np)
            for r in range(points.shape[0]):
                if not inpoly[r]:
                    outer_for_i.append(i)
                    if len(outer_for_i) > 10:
                        break
            outer += outer_for_i
        p =  points[outer]

        s = self.holepts.shape[0]
        v = self.holepts[:]
        w = self.holepts[list(range(1, s)) + [0]]
        return euclidean_projection(p, v, w).min(-1)[0], intersections, outer

    def inside_constraint(self, points):
        inpoly = parallelpointinpolygon(points.detach().numpy(), self.poly_np)
        # inside = [i for i, p in enumerate(points)
        #          if self.poly.oriented_side(sg.Point2(p[0], p[1])) == sg.Sign.POSITIVE]
        p = points[inpoly]

        s = self.holepts.shape[0]
        v = self.holepts[:]
        w = self.holepts[list(range(1, s)) + [0]]
        return euclidean_projection(p, v, w).min(-1)[0], p, inpoly

    def objective(self, p):
        constraint = p.view(self.npts, 1, 2) - self.holepts.view(1, -1, 2)
        constraint = constraint * constraint
        return constraint.min(-1)[0].sum()

    
    def dislikes(self, holes, pts):
      pts = pts.view(-1, 1, 2)
      holes = holes.view(1, -1, 2)
      d = dist(pts, holes).min(0).values
      return d.sum().item()

    def solve(self, starting_params, debug=False, mcmc=False):
        #parameters = starting_params.clone()
        #parameters.requires_grad_(True)
        parameter_struct = Params(starting_params)

        rate = 0.3
        #opt = torch.optim.SGD([parameters], lr=rate)
        opt = torch.optim.SGD(parameter_struct.raw_parameters(), lr=rate)

        success = False
        best_parameters = None
        best_dislike = float('inf')

        total_epochs = 4000
        for epochs in range(total_epochs):
            parameters = parameter_struct.get_parameters()

            #if best_parameters is not None and epochs >= 8000:
            #  break
            if debug and ((epochs % 1000) == 0 or epochs == 999 or (success and False)):
                plt.clf()
                draw(self.poly)
                vertices = [sg.Point2(a[0], a[1]) for a in parameters.detach().numpy()]
                edge_segments = [sg.Segment2(vertices[fro], vertices[to])
                                 for (fro, to) in self.graph]
                for segment in edge_segments:
                    draw(segment)
                plt.savefig("output%d.%d.png"%(self.problem_number, epochs))
                # __st.pyplot()
                if success == True and False:
                    plt.savefig("output%d.sol.%d.png"%(self.problem_number, epochs))
                    return {"vertices" : [[int(t[0].item()), int(t[1].item())] for t in p]}
            success = False
            opt.zero_grad()
            spring_cons = self.spring_constraint(parameters).mean()
            st_cons = self.stretch_constraint(parameters).sum()
            outside_cons, _, out_points = self.outside_constraint(parameters)
            random_cons, intersections, outer = self.random_constraint(parameters)
            inside_cons, _, _ = self.inside_constraint(parameters)
            inside_cons = inside_cons.clamp(max=12)
            int_cons = (parameters - parameters.round()).abs().mean()
            # loss = objective(parameters) + st_cons + outside_cons

            #if epochs < 5000:
            loss = -0.05 * inside_cons.sum()  + 0.1 * outside_cons.sum() +  spring_cons  + st_cons  + random_cons.sum()
            # else:
            #     loss =   0.1 * outside_cons.sum() + spring_cons
            loss.backward()
            #delta = (rate * parameters.grad)
            #parameters.data -= delta
            parameter_struct.update(rate) # update bias as well for global translation

            # regenerate after taking a step
            parameters = parameter_struct.get_parameters()

            if True:
                # opt.lr = rate * 0.5
                decay = 0.02
                roundies = (parameters.data - parameters.data.round()).abs() <= decay
                parameters.data[roundies] =  parameters.data[roundies].round()
                noroundies = (parameters.data - parameters.data.round()).abs() > decay
                parameters.data[noroundies] -=  decay * torch.sign(parameters.data[noroundies] - parameters.data[noroundies].round())

            # update parameter_struct with rounded params
            parameter_struct.set_parameters(parameters)

            p = parameters.detach().round().float()
            
            if self.stretch_constraint(p).sum().item() == 0.0 and self.outside_constraint(p)[0].sum().item() == 0.0 and out_points.sum() == 0:
                _, intersections, _ = self.random_constraint(p)
                if len(intersections) == 0:
                    print("success!")
                    #print({"vertices" : [[int(t[0].item()), int(t[1].item())] for t in p]})
                    success = True

                    dislike = self.dislikes(self.holepts, parameters)
                    if dislike < best_dislike:
                      best_dislike = dislike
                      best_parameters = p.data.clone()

            if epochs % 100 == 0.0:

                p = parameters.round()        
                d = {"round" : epochs,
                     "best_dislike": best_dislike,
                     "loss": loss.detach().item(),
                     "spring" : spring_cons.detach().item(),
                     "intersections" : intersections,
                     "out_points" : out_points.sum(),
                     "stretch" : st_cons.detach().item(),
                     "outside" : outside_cons.sum().detach().item(),
                     "int cons" : int_cons.item(),
                     "round stretch": self.stretch_constraint(p).sum().item(),
                     "round outside": self.outside_constraint(p)[0].sum().item()
                     # "int con" : int_cons.detach().item()
                }
                print(d)
        if mcmc:
          # simulated annealing
          import math
          import random
          min_dislike = float('inf')
          best_ps = None

          eps = 1e-4
          def temperature(e, es):
            T = 1
            return T * (1 - e / es)

          def proposal(p, state):
            p = p.data.clone()
            state = state % p.shape[0]
            new_pos_delta_possible = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
            new_pos_delta = random.choice(new_pos_delta_possible)
            p[state, 0] += new_pos_delta[0]
            p[state, 1] += new_pos_delta[1]
            return p, state+1

          def energy(p):
            stretch = self.stretch_constraint(p).sum().item()
            outside = self.outside_constraint(p)[0].sum().item()
            dislike = self.dislikes(self.holepts, p)
            intersections = self.find_intersections(p)
            E = stretch * 10 + outside * 10 + dislike/100 + len(intersections) * 100
            return E

          if best_parameters is None:
            best_parameters = p
          p_sa = best_parameters.data.clone()
          E = energy(p_sa)
          print (E)
          print ({"round stretch": self.stretch_constraint(p_sa).sum().item(),
          "round outside": self.outside_constraint(p_sa)[0].sum().item(),
          "intersections": len(self.find_intersections(p_sa)),
          "dislike": self.dislikes(self.holepts, p_sa)})
          epochs = 8000
          state = 0
          accepted = 0
          total = 0
          for epoch in range(epochs):
            for _ in range(p.shape[0]):
              t = temperature(epoch*p.shape[0], epochs*p.shape[0])
              p_sa_prop, state = proposal(p_sa, state)
              E_prop = energy(p_sa_prop)
              #print ((E - E_prop) / max(t, eps))
              acceptance_ratio = math.exp(min(0, (E - E_prop) / max(t, eps)))
              total += 1
              if random.random() < acceptance_ratio:
                #assert E > E_prop
                #print (p_sa_prop - p_sa, E, E_prop, acceptance_ratio)
                accepted += 1
                p_sa = p_sa_prop
                E = E_prop
                if self.stretch_constraint(p_sa).sum().item() == 0.0 and self.outside_constraint(p_sa)[0].sum().item() == 0.0 and len(self.find_intersections(p_sa))==0:
                  print("success!")
                  print(p)
                  print({"vertices" : [[int(t[0].item()), int(t[1].item())] for t in p]})
                  success = True
                  dislike  = self.dislikes(self.holepts, p_sa)
                  if dislike < min_dislike:
                    min_dislike = dislike
                    best_ps = p_sa.data.clone()
            if epoch % 100 == 0:
              print (f'epoch: {epoch}, E: {E}, accepted: {accepted}, total: {total}')
              print ({"round stretch": self.stretch_constraint(p_sa).sum().item(),
          "round outside": self.outside_constraint(p_sa)[0].sum().item(),
          "intersections": len(self.find_intersections(p_sa)),
          "dislike": self.dislikes(self.holepts, p_sa)})

          print (min_dislike)
          print (best_ps)
          best_parameters = best_ps
        if best_parameters is not None:
          plt.clf()
          draw(self.poly)
          vertices = [sg.Point2(a[0], a[1]) for a in best_parameters.detach().numpy()]
          edge_segments = [sg.Segment2(vertices[fro], vertices[to])
                            for (fro, to) in self.graph]
          for segment in edge_segments:
              draw(segment)
          plt.savefig("output%d.sol.%d.png"%(self.problem_number, epochs))
          return {"vertices" : [[int(t[0].item()), int(t[1].item())] for t in best_parameters]}
        return None
SUBMIT = False
for problem_number in range(2, 3):
    problem = Problem(problem_number)
    # result = problem.solve(torch.rand(*problem.original.shape), debug = True)
    result = problem.solve(problem.original, debug = True, mcmc=True)
    if result is not None:
        with open("p%d.sol.json"%problem_number, "w") as w:
            w.write(json.dumps(result))
        if SUBMIT:
            import requests
            # api-endpoint
            URL = f"https://poses.live/api/problems/{problem_number}/solutions"

            # defining a params dict for the parameters to be sent to the API
            PARAMS = {'Authorization': 'Bearer eb72d58e-adcf-4d47-9853-6c4680de6ffe'}
            
            # sending get request and saving the response as response object
            r = requests.post(url = URL, json=result, headers = PARAMS)
            print (r)
            # extracting data in json format
            data = r.json()
              
            print (data)
