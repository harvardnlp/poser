# ICFP Contest
import pandas as pd
import skgeom as sg
from skgeom.draw import draw
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
# __st.set_option('deprecation.showPyplotGlobalUse', False)
from polygons import *
import polygons
import math
import random
import time
from matplotlib.path import Path
import itertools
        
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
    def __init__(self, starting_params, nochange=None):
        self.parameters = starting_params.clone()
        self.parameters.requires_grad_(True)
        # x and y bias
        self.bias = torch.zeros(2)
        self.bias.requires_grad_(True)
        self.nochange = nochange
        
    def get_parameters(self):
        return self.parameters + self.bias

    def update(self, rate):
        if self.nochange is not None:
            self.parameters.grad[nochange] = 0.0
            self.parameters.data -= rate * self.parameters.grad
        else:
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


        filename = "p%d.json"%problem_number
        if not os.path.exists(filename):
            os.system(f"wget https://poses.live/problems/{problem_number}/download -O p{problem_number}.json")
        poses = json.load(open(filename))

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
        self.target = float(self.epsilon) / float(1000000.0)
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

        self.edge_exists = {}
        graph_set = set([(g[0].item(), g[1].item()) for g in self.graph])
        print(graph_set)
        for i in range(len(self.vertices)):
            for j in range(len(self.vertices)):
                if (i, j) in graph_set or (j, i) in graph_set:
                    self.edge_exists[i, j] = dist(self.original[i], self.original[j])
                    self.edge_exists[j, i] = dist(self.original[i], self.original[j])

        # print(graph_set, self.edge_exists)
        self.hole_exists = {}
        for i in range(self.holepts.shape[0]):
            for j in range(self.holepts.shape[0]):
                self.hole_exists[i, j] =  dist(self.holepts[i], self.holepts[j])
                

                
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
            
    def guessandcheck(self):
        import itertools
        total = len(self.vertices)
        holes = self.holepts.shape[0]
        compat = torch.zeros(total, total, holes, holes).bool()
        for o in range(total):
            for o2 in range(total):
                for i in range(holes):
                    for j in range(holes):
                        if (o, o2) in self.edge_exists:
                            compat[o, o2, i, j] = (torch.abs((self.edge_exists[o, o2] / self.hole_exists[i,j]) - 1.0) <= self.target)

        for order in itertools.product(range(holes), repeat=total):
            fail = False
            for o, i in enumerate(order):
                for o2, j in enumerate(order[o+1:], o+1):
                    if (o, o2) in self.edge_exists and not compat[o, o2, i, j]:
                        fail = True
                        
            if fail:
                continue
            print("good", order)
            p = torch.tensor([self.holepts[o].tolist() for o in order])
            # if self.stretch_constraint(p).sum() < 1.0:
            #     print(order, self.stretch_constraint(p), self.seg_dist(p), self.comp, target)
            # if self.stretch_constraint(p).sum().item() == 0.0:
            
                # if self.outside_constraint(p)[0].sum().item() == 0.0:
            _, intersections, _ = self.random_constraint(p)
            if len(intersections) == 0:
                return {"vertices" : [[int(t[0].item()), int(t[1].item())] for t in p]}
            # else:
            #     print("bad")
                    
                #     else:
                #         print("failed intersection")
                # else:
                #     print("failed outside")
            
    def guessandcheck2(self, starting_params):

        total = len(self.vertices)
        holes = self.holepts.shape[0]

        compat = torch.zeros(total, total, holes, holes).bool()
        compat[:] = True
        for o in range(total):
            for o2 in range(o+1, total):
                for i in range(holes):
                    for j in range(i+1, holes):
                        if (o, o2) in self.edge_exists and i != j:
                            compat[o, o2, i, j] = (torch.abs((self.edge_exists[o, o2] / self.hole_exists[i,j]) - 1.0) <= self.target)
                            if not compat[o, o2, i, j]:
                                print(o, o2, i, j)
        print("hello", list(range(total)), holes)
        for order in itertools.permutations(list(range(total)), r=holes):
            fail = False
            print(order)
            for i, o in enumerate(order):
                for j, o2 in enumerate(order[i+1:], i+1):
                    if not compat[o, o2, i, j]:
                        fail = True
                        
            if not fail:
                for i, o in enumerate(order):
                    starting_params[o] = self.holepts[i]
                nochange = torch.zeros(total).bool()
                for o in order:
                    nochange[o] = True
                yield starting_params, nochange
            # p = torch.tensor([self.holepts[o].tolist() for o in order])
            # # if self.stretch_constraint(p).sum() < 1.0:
            # #     print(order, self.stretch_constraint(p), self.seg_dist(p), self.comp, target)
            # if self.stretch_constraint(p).sum().item() == 0.0:
            #     if self.outside_constraint(p)[0].sum().item() == 0.0:
            #         _, intersections, _ = self.random_constraint(p)
            #         if len(intersections) == 0:
            #             return {"vertices" : [[int(t[0].item()), int(t[1].item())] for t in p]}
        print("done")
            
    def solve(self, starting_params, nochange=None, debug=False, mcmc=False, epochs=8000, mcmc_epochs=8000):

        if nochange is not None and nochange.all():
            return {"vertices" : [[int(t[0].item()), int(t[1].item())] for t in starting_params]}
        #parameters = starting_params.clone()
        #parameters.requires_grad_(True)
        parameter_struct = Params(starting_params, nochange)

        rate = 0.3
        #opt = torch.optim.SGD([parameters], lr=rate)
        opt = torch.optim.SGD(parameter_struct.raw_parameters(), lr=rate)

        success = False
        best_parameters = None
        best_dislike = float('inf')

        total_epochs = epochs
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
            dislike = self.dislikes(self.holepts, parameters)
            if epochs < 4000:
                loss = -0.1 * inside_cons.sum()  + 1.0 * outside_cons.sum() +  2*spring_cons  + st_cons + random_cons.sum()
            else:
                loss = 1.0 * outside_cons.sum() + spring_cons
            # if random_cons.shape[0] > 0: 
            #     loss +=  random_cons.sum()
            # else:
            #     loss =   0.1 * outside_cons.sum() + spring_cons
            loss.backward()
            #delta = (rate * parameters.grad)
            #parameters.data -= delta
            if epochs > 0: 
                parameter_struct.update(rate) # update bias as well for global translation

                # regenerate after taking a step
                parameters = parameter_struct.get_parameters()

                if epochs > 4000:
                    # opt.lr = rate * 0.5
                    decay = 0.2
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
                    # print("success!")
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
                     "dislike":dislike,
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

            action = random.choices(
                ["vertex_translate", "global_translate", "global_rotate"],
                weights = [10, 0, 0],
            )[0]

            if action == "vertex_translate":
                # perturb one vertex
                local = True
                new_pos_delta_possible = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
                new_pos_delta = random.choice(new_pos_delta_possible)

                if nochange is not None and  nochange[state]:
                    pass
                else:
                    p[state, 0] += new_pos_delta[0]
                    p[state, 1] += new_pos_delta[1]

                state = state + 1

            elif action == "global_translate":
                # perturb global
                local = False
                new_pos_delta_possible = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
                new_pos_delta = random.choice(new_pos_delta_possible)

                p[:, 0] += new_pos_delta[0]
                p[:, 1] += new_pos_delta[1]

            elif action == "global_rotate":
                # rotate global
                local = False
                angle = random.choice([45, 90, 135, 180, 225, 270, 315])
                s = math.sin(angle)
                c = math.cos(angle)

                # rotate every point angle degrees around center
                center = p[state]

                p_centered = p - center
                p[:,0] = c * p_centered[:,0] - s * p_centered[:,0] + center[0] 
                p[:,1] = c * p_centered[:,1] + s * p_centered[:,1] + center[1]
                p = p.round()
            else:
                raise ValueError(f"Wrong action {action}")

            return p, state, local 


          def energy_state(p):              
              inpoly = parallelpointinpolygon(p.detach().numpy(), self.poly_path)
              return {"inpoly" : inpoly,
                      "stretch" : self.spring_constraint(p),
                      "outside" : self.outside_constraint(p)[0],
                      "dislike" : self.dislikes(self.holepts, p),
                      "intersections" : self.find_intersections(p, inpoly)}

          def update_energy_state(p, p_old, state, cache):
              d =  {"inpoly" : np.array(cache["inpoly"]),
                    
                    "outside" : cache["outside"].clone(),
                    "dislike" : self.dislikes(self.holepts, p)}
              d["stretch"] = self.spring_constraint(p)
              d["inpoly"][state:state+1] = parallelpointinpolygon(p[state:state+1], self.poly_path)
              new, old = self.find_intersections_dyn(p, d["inpoly"], state)
              d["intersections"] = set(cache["intersections"]) - set(old) | set(new)
              d["outside"] = self.outside_constraint(p, d["inpoly"])[0]
              
              return d
          
          def energy(p, p_old=None, cache=None, state=None):
          
            if cache is None:
                state_cache = energy_state(p)
            else:
                state = state % p.shape[0]
                state_cache = update_energy_state(p, p_old, state, cache)
            E = state_cache["stretch"].sum().item()+ 10 * state_cache["outside"].sum().item() + (state_cache["dislike"]/100000) + len(state_cache["intersections"]) * 100
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
          epochs = mcmc_epochs
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
              p_sa_prop, state, local = proposal(p_sa, state)              
              E_prop, E_cache_prop = energy(p_sa_prop, p_sa, E_cache if local else None, state - 1)
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
                  # print("success!")
                  # print(p)
                  # print({"vertices" : [[int(t[0].item()), int(t[1].item())] for t in p]})
                  success = True
                  dislike  = self.dislikes(self.holepts, p_sa)
                  if dislike < min_dislike:
                    min_dislike = dislike
                    print("Dislike down to ", min_dislike)
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
SUBMIT = True
for problem_number in range(31, 32):
    problem = Problem(problem_number)
    # result = problem.solve(torch.rand(*problem.original.shape), debug = True, mcmc=True)
    # result = problem.solve(problem.original, debug = False, mcmc=True)

    if True:
        result = problem.solve(problem.original, debug = False, mcmc=True)
    elif False:
        result = problem.guessandcheck()
    else:
        result = None
        for start, nochange in problem.guessandcheck2(torch.rand(*problem.original.shape)):
            print(nochange)
            result = problem.solve(start, debug = False, mcmc=True, mcmc_epochs=1000, epochs=2000, nochange=nochange)
            if result is not None:
                break
    if result is not None:
        with open("p%d.sol.json"%problem_number, "w") as w:
            w.write(json.dumps(result))
        # with open("p%d.sol.json"%problem_number, "r") as r:
        #     result = json.loads(r.read())

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
