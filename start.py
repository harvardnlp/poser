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
import math
import random
import time
from matplotlib.path import Path

def dist(u, v):
    d = u - v
    return (d * d).sum(-1)

def euclidean_projection(p, v, w):
    p = p.view(-1, 1, 2)
    l2 = dist(v, w)
    t = torch.clamp(torch.einsum("psd,sd->ps", p - v, w - v) / l2, max=1, min=0)
    projection = v + t[..., None] * (w - v)
    d = torch.linalg.norm((p-projection))
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
        self.comp = self.seg_dist(self.original)
        s = self.holepts.shape[0]
        self.v = self.holepts[:].view(-1, 2)
        self.w = self.holepts[list(range(1, s)) + [0]].view(-1, 2)
        self.b1 = torch.tensor(self.holepts[:]).float()
        self.b2 = torch.tensor(self.holepts[list(range(1, s)) + [0]]).float()
        self.poly_path = Path(self.poly_np)

        self.edges = {}
        for i in range(len(self.vertices)):
            self.edges[i] = set([j for j, (fro, to) in enumerate(self.graph)
                                 if to == i or fro == i])
        
    def find_intersections(self, p, inpoly=None, debug=False):
        if inpoly is None:
            inpoly = parallelpointinpolygon(p.detach().numpy(), self.poly_path)
        bad_edges = []
        s = self.holepts.shape[0]
        p = p.detach()
        edge_segments = [(i, p[fro], p[to])
                         for i, (fro, to) in enumerate(self.graph)
                         if inpoly[fro] and inpoly[to]]

        a1 = torch.tensor([[v[1][0], v[1][1]] for v in edge_segments]).float()
        a2 = torch.tensor([[v[2][0], v[2][1]] for v in edge_segments]).float()
        n = seg_intersect(a1, a2, self.b1, self.b2)

        # if debug:
        #     for i in range(n.shape[0]):
        #         for j in range(n.shape[1]):
        #             if n[i, j]:
        #                 print("intersect", a1[i], a2[i], b1[j], b2[j])
        bad = n.any(dim=1)
        for i in range(bad.shape[0]):
            if bad[i]:
                bad_edges.append(edge_segments[i][0])
        return bad_edges

    def find_intersections_dyn(self, p, inpoly=None, changedpt=None, debug=False):
        if inpoly is None:
            inpoly = parallelpointinpolygon(p.detach().numpy(), self.poly_path)
        bad_edges = []
        s = self.holepts.shape[0]
        p = p.detach()
        total = self.edges[changedpt]
        edge_segments = [(i, p[fro], p[to])
                         for i in self.edges[changedpt]
                         for (fro, to) in [self.graph[i]]
                         if inpoly[fro] and inpoly[to]]

        a1 = torch.tensor([[v[1][0], v[1][1]] for v in edge_segments]).float()
        a2 = torch.tensor([[v[2][0], v[2][1]] for v in edge_segments]).float()
        n = seg_intersect(a1, a2, self.b1, self.b2)

        # if debug:
        #     for i in range(n.shape[0]):
        #         for j in range(n.shape[1]):
        #             if n[i, j]:
        #                 print("intersect", a1[i], a2[i], b1[j], b2[j])
        bad = n.any(dim=1)
        for i in range(bad.shape[0]):
            if bad[i]:
                bad_edges.append(edge_segments[i][0])
        return bad_edges, total

    def seg_dist(self, p):
        constraint = p[self.graph[:, 0]] - p[self.graph[:, 1]]
        return (constraint * constraint).sum(-1)

    # The main constraint

    def stretch_constraint(self, p):
        return torch.relu(((self.seg_dist(p) / self.comp) - 1.0).abs() - self.target)

    def spring_constraint(self, p):
        d = (self.seg_dist(p) - self.seg_dist(self.original))
        return d.abs()

    def outside_constraint(self, points, inpoly=None):
        if inpoly is None:
            inpoly = parallelpointinpolygon(points.detach().numpy(), self.poly_path)

        p = points[~inpoly]
        d = euclidean_projection(p, self.v, self.w).min(-1)[0]
        return d, p, ~inpoly

    # def outside_constraint_dyn(self, points):
    #     inpoly = parallelpointinpolygon(points.detach().numpy(), self.poly_path)
    #     p = points
    #     d = euclidean_projection(p, self.v, self.w).min(-1)[0]
    #     d[inpoly] = 0.0
    #     return d, p, ~inpoly

    def random_constraint(self, points):
        inpoly = parallelpointinpolygon(points.detach().numpy(), self.poly_path)
        intersections = self.find_intersections(points, inpoly)
        new_points = []
        r = torch.rand(100, 1, 1)
        points = r * points[self.graph[intersections, 0]]  + (1-r) * points[self.graph[intersections,1]]

        outer = []
        for i in range(points.shape[1]):
            outer_for_i = []
            inpoly = parallelpointinpolygon(points[:, i].detach().numpy(), self.poly_path)
            for r in range(points.shape[0]):
                if not inpoly[r]:
                    outer_for_i.append(i)
                    if len(outer_for_i) > 10:
                        break
            outer += outer_for_i
        p =  points[outer]
        return euclidean_projection(p, self.v, self.w).min(-1)[0], intersections, outer

    def inside_constraint(self, points):
        inpoly = parallelpointinpolygon(points.detach().numpy(), self.poly_path)
        p = points[inpoly]
        d = euclidean_projection(p, self.v, self.w).min(-1)[0]
        return d, p, ~inpoly

        # d = euclidean_projection(points, self.v, self.w).min(-1)[0]
        # d[~inpoly] = 0.0

        # # inside = [i for i, p in enumerate(points)
        # #          if self.poly.oriented_side(sg.Point2(p[0], p[1])) == sg.Sign.POSITIVE]
        # return d, points, inpoly

    def objective(self, p):
        constraint = p.view(self.npts, 1, 2) - self.holepts.view(1, -1, 2)
        constraint = constraint * constraint
        return constraint.min(-1)[0].sum()

    
    def dislikes(self, holes, pts):
      pts = pts.view(-1, 1, 2)
      holes = holes.view(1, -1, 2)
      d = dist(pts, holes).min(0).values
      return d.sum().item()

  
    def show(self, params, save=""):
        plt.clf()
        draw(self.poly)
        violations = self.stretch_constraint(params) > 0.0
        # self.find_intersections(params.detach(), debug=True)
        random_cons, intersections, outer = self.random_constraint(params.detach())
        outside_cons, _, out_points = self.outside_constraint(params.detach())
        intersections = set(intersections)
        vertices = [sg.Point2(a[0], a[1]) for a in params.detach().numpy()]
        edge_segments = [sg.Segment2(vertices[fro], vertices[to])
                         for (fro, to) in self.graph]
        for i, segment in enumerate(edge_segments):
            if i in intersections:
                draw(segment, color="red")
            elif violations[i]:
                draw(segment, color="yellow")
            else:
                draw(segment, color="blue")
        for i, vertex in enumerate(vertices):
            if out_points[i]:
                draw(vertex, color="red")
            else:
                draw(vertex, color="black")
        if save:
            plt.savefig(save)

  
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
            if debug and ((epochs % 1000) == 0 or epochs == 999):
                self.show(parameters,
                          save="output%d.%d.png"%(self.problem_number, epochs))
                self.show(parameters.round(),
                          save="output%d.%d.int.png"%(self.problem_number, epochs))

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
            # print(inside_cons.mean(), outside_cons.mean(), random_cons.mean())
            # loss = 1.0 * outside_cons.sum()  -0.05 * inside_cons.sum()
            loss = -0.05 * inside_cons.sum()  + 1.0 * outside_cons.sum() +  spring_cons  + st_cons + random_cons.sum()
            # if random_cons.shape[0] > 0: 
            #     loss +=  random_cons.sum()
            # else:
            #     loss =   0.1 * outside_cons.sum() + spring_cons
            loss.backward()
            #delta = (rate * parameters.grad)
            #parameters.data -= delta
            parameter_struct.update(rate) # update bias as well for global translation

            # regenerate after taking a step
            parameters = parameter_struct.get_parameters()

            if False:
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
          min_dislike = float('inf')
          best_ps = None

          eps = 1e-4
          def temperature(e, es):
            T = 1
            return T * (1 - e / es)

          def proposal(p, state):
            p = p.data.clone()
            state = state % p.shape[0]
            new_pos_delta_possible = [[-1, 0], [1, 0], [0, -1], [0, 1],
                                      [1, 1], [1, -1], [-1, 1], [-1, -1]]
            new_pos_delta = random.choice(new_pos_delta_possible)
            p[state, 0] += new_pos_delta[0]
            p[state, 1] += new_pos_delta[1]
            return p, state+1

          def energy_state(p):              
              inpoly = parallelpointinpolygon(p.detach().numpy(), self.poly_path)
              return {"inpoly" : inpoly,
                      "stretch" : self.stretch_constraint(p),
                      "outside" : self.outside_constraint(p)[0],
                      "dislike" : self.dislikes(self.holepts, p),
                      "intersections" : self.find_intersections(p, inpoly)}

          def update_energy_state(p, p_old, state, cache):
              d =  {"inpoly" : np.array(cache["inpoly"]),
                    
                    "outside" : cache["outside"].clone(),
                    "dislike" : self.dislikes(self.holepts, p)}
              # start = time.time()
              d["stretch"] = self.stretch_constraint(p)
              # end = time.time()
              # print("a", end-start)
              # start = time.time()
              # d["dislike"] -= self.dislikes(self.holepts, p_old[state:state+1])
              # d["dislike"] += self.dislikes(self.holepts, p[state:state+1])
              # end = time.time()
              # print("b", end-start)
              # start = time.time()
              d["inpoly"][state:state+1] = parallelpointinpolygon(p[state:state+1], self.poly_path)
              new, old = self.find_intersections_dyn(p, d["inpoly"], state)
              d["intersections"] = set(cache["intersections"]) - set(old) | set(new)
              # end = time.time()
              # print("c", end-start)
              # start = time.time()
              d["outside"] = self.outside_constraint(p, d["inpoly"])[0]
              # end = time.time()
              # print("d", end-start)
              
              return d
          
          def energy(p, p_old=None, cache=None, state=None):
            # stretch = self.stretch_constraint(p).sum().item()
            # outside = self.outside_consrtaint(p)[0].sum().item()
            # dislike = self.dislikes(self.holepts, p)
            # intersections = self.find_intersections(p)
          
            if cache is None:
                state_cache = energy_state(p)
            else:
                state = state % p.shape[0]
                state_cache = update_energy_state(p, p_old, state, cache)
                # state_cache = energy_state(p)
                # print("dyn", state_cache)
                # print("reg", energy_state(p))
                # E2 = state_cache["stretch"].sum().item() * 10 + state_cache["outside"].sum().item() * 10 + state_cache["dislike"]/100 + len(state_cache["intersections"]) * 100
                # # print("E1", E, state_cache["dislike"], len(state_cache["intersections"]))
                # state_cache = energy_state(p)
                # E = state_cache["stretch"].sum().item() * 10 + state_cache["outside"].sum().item() * 10 + state_cache["dislike"]/100 + len(state_cache["intersections"]) * 100
                # assert E == E2
            # print(state_cache["stretch"].sum().item() * 10, state_cache["outside"].sum().item() * 10, + state_cache["dislike"]/100,  len(state_cache["intersections"]) * 100)
            E = state_cache["stretch"].sum().item() * 10 + state_cache["outside"].sum().item() * 10 + state_cache["dislike"]/100 + len(state_cache["intersections"]) * 100
            # print("E2", E, state_cache["dislike"], len(state_cache["intersections"]))
            return E, state_cache

          if best_parameters is None:
            best_parameters = p
          p_sa = best_parameters.detach().clone()
          E, E_cache = energy(p_sa)
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
            if debug and (epoch % 500) == 0 :
                print("showing")
                self.show(best_parameters.detach(),
                          save="output%d.mcmc.%d.png"%(self.problem_number, epoch))
                print("done")
              
            for _ in range(p.shape[0]):
              t = temperature(epoch*p.shape[0], epochs*p.shape[0])
              p_sa_prop, state = proposal(p_sa, state)              
              E_prop, E_cache_prop = energy(p_sa_prop, p_sa, E_cache, state - 1)
              #print ((E - E_prop) / max(t, eps))
              acceptance_ratio = math.exp(min(0, (E - E_prop) / max(t, eps)))
              total += 1
              if random.random() < acceptance_ratio:
                #assert E > E_prop
                #print (p_sa_prop - p_sa, E, E_prop, acceptance_ratio)
                accepted += 1
                p_sa = p_sa_prop
                E = E_prop
                E_cache = E_cache_prop
                if E_cache["stretch"].sum().item() == 0.0 and E_cache["outside"].sum().item() == 0.0 and len(E_cache["intersections"])==0:
                  print("success!")
                  print(p)
                  print({"vertices" : [[int(t[0].item()), int(t[1].item())] for t in p]})
                  success = True
                  dislike  = self.dislikes(self.holepts, p_sa)
                  if dislike < min_dislike:
                    min_dislike = dislike
                    best_ps = p_sa.data.clone()
            if epoch % 100 == 0:
              print (f'mcmc epoch: {epoch}, E: {E}, accepted: {accepted}, total: {total}')
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
for problem_number in range(5, 6):
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
