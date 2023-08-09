"""
Find minimum energy path (MEP) along nearest neighbor graph
"""
import argparse
import os
import shutil
import pickle
from heapq import heappop, heappush

import numpy as np
import torch


def add_args(parser):
    parser.add_argument("data", help="Input latent space z")
    parser.add_argument(
        "--anchors", type=int, nargs="+", required=True, help="Index of anchor points"
    )
    parser.add_argument("--n-neighbors", type=int, default=50, help="The number of neighbors to construt a graph")
    parser.add_argument(
        "--search-q", type=float, nargs="+", help="0.1,...,0.9 will be applied if not specified."
    )
    parser.add_argument(
        "--zero-ratio-l", type=float, default=0.01, help="The minimum zero energy ratio to control the shape of energy landscape."
    )
    parser.add_argument(
        "--zero-ratio-h", type=float, default=0.1, help="The maximum zero energy ratio to control the shape of energy landscape."
    )
    parser.add_argument(
        "--output-all", action='store_true', help="Output all paths reach the maximum zero energy ratio" 
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "-o",
        "--outdir",
        type=os.path.abspath,
        required=True,
        help="Output directory to save results",
    )
    return parser


class Graph(object):
    def __init__(self, edges):  # edges is a list of tuples (src, dest, distance)
        # everything after here is derived from (weights, actions, probs)
        # for computational efficiency

        # FIXME: nodes and goal nodes should be the same
        self.nodes = set([x[0] for x in edges] + [x[1] for x in edges])
        self.edges = {x: set() for x in self.nodes}
        self.edge_length = {}
        for s, d, L in edges:
            assert type(s) == int and type(d) == int and type(L) == float
            self.edges[s].add(d)
            self.edge_length[(s, d)] = L

    def find_path(self, src, dest):
        visited = set()
        unvisited = []
        distances = {}
        predecessors = {}

        distances[src] = 0
        heappush(unvisited, (0, src))

        while unvisited:
            # visit the neighbors
            dist, v = heappop(unvisited)
            if v in visited or v not in self.edges:
                continue
            visited.add(v)
            if v == dest:
                # We build the shortest path and display it
                path = []
                pred = v
                while pred is not None:
                    path.append(pred)
                    pred = predecessors.get(pred, None)
                return path[::-1], dist

            neighbors = list(self.edges[v])

            for idx, neighbor in enumerate(neighbors):
                if neighbor not in visited:
                    new_distance = dist + self.edge_length[(v, neighbor)]
                    if new_distance < distances.get(neighbor, float("inf")):
                        distances[neighbor] = new_distance
                        heappush(unvisited, (new_distance, neighbor))
                        predecessors[neighbor] = v

        # couldn't find a path
        return None, None


def main(args):
    data_np = pickle.load(open(args.data, "rb"))
    data = torch.from_numpy(data_np)

    use_cuda = torch.cuda.is_available()
    print(f"Use cuda {use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")
    data = data.to(device)

    N, D = data.shape
    for i in args.anchors:
        assert i < N
    assert len(args.anchors) >= 2

    n2 = (data * data).sum(-1, keepdim=True)
    B = args.batch_size
    ndist = torch.empty(data.shape[0], args.n_neighbors, device=device)
    neighbors = torch.empty(
        data.shape[0], args.n_neighbors, dtype=torch.long, device=device
    )
    for i in range(0, data.shape[0], B):
        # (a-b)^2 = a^2 + b^2 - 2ab
        # print(f"Working on images {i}-{i+B}")
        batch_dist = n2[i : i + B] + n2.t() - 2 * torch.mm(data[i : i + B], data.t())
        ndist[i : i + B], neighbors[i : i + B] = batch_dist.topk(
            args.n_neighbors, dim=-1, largest=False
        )

    assert ndist.min() >= -1e-3, ndist.min()
    # convert d^2 to d
    ndist = ndist.clamp(min=0).pow(0.5)

    # specified searching quantiles 
    if args.search_q is not None:
        search_quantiles = args.search_q
    else:
        search_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print()
    print(f"Searching in: {search_quantiles}")
    print(f"The range of zero energy ratio: ({args.zero_ratio_l}, {args.zero_ratio_h})")
    print()

    density_metric = float("inf") # metric for choosing the best path
    for q in search_quantiles:

        # threshold choose by quantiles
        max_dist = torch.quantile(ndist, q=q)

        # calculate free energy = -log(n_i/n_max)
        grids_counts = (ndist <= max_dist).sum(dim=1)
        eng = -torch.log(grids_counts/grids_counts.max())
        eng[eng==-0.] = 0.

        states, counts = torch.unique(eng, return_counts=True)
        zero_energy_ratio = counts[0]/N
        if states.shape[0] <2 or  zero_energy_ratio<args.zero_ratio_l or zero_energy_ratio>args.zero_ratio_h:
            print()
            print(f'Quantiles={q} are not suitable.')
            continue
        else:
            print()
            print(f"Construt graph with quantile={q}")
            print("...Calculating free-energy...")
            eng_dist = torch.empty(N, args.n_neighbors, dtype=eng.dtype)
            for i in range(N):
                neighbor_indices = neighbors[i]
                eng_dist[i] = (eng[neighbor_indices]+eng[i])/2.
            eng_dist = eng_dist.to("cpu")
            neighbors = neighbors.to("cpu")
            # ndist = ndist.to("cpu")
            eng = eng.to("cpu")

            edges_eng = []
            for i in range(neighbors.shape[0]):
                for j in range(neighbors.shape[1]):
                    edges_eng.append((int(i), int(neighbors[i, j]), float(eng_dist[i,j])))

            # searching
            print("...Searching MEP...")
            graph = Graph(edges_eng)
            full_path = []
            for i in range(len(args.anchors)-1):
                src, dest = args.anchors[i], args.anchors[i+1]
                path, total_energy = graph.find_path(src, dest)

                print("Path:")
                for id in path:
                    print(id)

                if path is not None:
                    if full_path and full_path[-1] == path[0]:
                        full_path.extend(path[1:])
                    else:
                        full_path.extend(path)
                else: 
                    full_path.append([None])

            if args.output_all:
                if args.outdir is not None and not os.path.exists(args.outdir):
                    os.makedirs(args.outdir)
                np.savetxt(f"{args.outdir}/Quantile_{q}.txt", full_path, fmt="%d")

            else:
                # 1/len(path)\sum(distance to neighbors/n)
                mean_below_threshold = (ndist * (ndist < max_dist).float()).sum(dim=1) / (ndist < max_dist).sum(dim=1).clamp(min=1)
                dens = torch.mean(mean_below_threshold[full_path]).to('cpu')
                print(f"Density metric: {dens}")
                if dens < density_metric:
                    best_q = q
                    best_path = full_path
                    density_metric = dens

    if os.path.exists(args.outdir):
        shutil.rmtree(args.outdir)
        os.makedirs(args.outdir)
    else:
        os.makedirs(args.outdir)
    np.savetxt(f"{args.outdir}/Best_q_{best_q}.txt".replace(".", ""), best_path, fmt="%d")    
 

parser = argparse.ArgumentParser(description=__doc__)
add_args(parser)
main(parser.parse_args())
