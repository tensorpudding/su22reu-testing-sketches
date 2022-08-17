#!/usr/bin/env python3

from l_zero_graph import L0Sketch
import numpy as np
import sys
import graph_tool as gt
import graph_tool.draw as draw

def main():
    if (len(sys.argv) > 1):
        seed = int(sys.argv[1])
    else:
        seed = 172
    #edges = [(1, 3, 1), (2, 5, 1), (1, 5, 1), (0, 9, 1), (3, 4, 1), (4, 9, 1)]
    #n = 3
    #edges = [(0, 1, 1), (0, 2, 1)]
    # CONNECTED EXAMPLE
    #edges = [(0,3,1),(1,2,1),(2,3,1),(2,4,1),(2,5,1),(4,6,1),(4,7,1),(5,6,1),(7,8,1),(7,9,1)]
    # DISCONNECTED EXAMPLE
    #edges = [(0,3,1),(1,2,1),(2,3,1),(2,5,1),(4,7,1),(5,6,1),(7,8,1),(7,9,1)]
    
    # Loading graph
    graph = gt.load_graph('github.gt')
    graph.set_directed(False)
    graph.ep["s"] = graph.new_edge_property("double", val=0.2)
    # Example code
    # in_graph= gt.Graph(directed=False)
    # in_graph.add_vertex(50)
    # in_graph.add_edge(0, 1)
    # in_graph.add_edge(2, 1)


    n = graph.num_vertices()
    NUM_SKETCHES=int(2*np.log(n))
    vec_sketches = []
    for i in range(NUM_SKETCHES):
        vec_sketches.append([])
    print(f"Using {NUM_SKETCHES} sketches for each of {n} vertices")
    rng = np.random.default_rng(seed=seed)
    first_seed = rng.integers(2**32)
    sketch_seeds = rng.integers(2**32, size=NUM_SKETCHES)
    #print(f"Testin", flush=True)
    for k in range(n):
        for phase in range(NUM_SKETCHES):
            vec_sketches[phase].append(L0Sketch(n, seed=sketch_seeds[phase]))
    for i,j in graph.iter_edges():
        for phase in range(NUM_SKETCHES):
            vec_sketches[phase][i].update(i,j,1)
            #print(f"Added ({i},{j}) to sketch S_[{phase}](a^{i})", flush=True)
            vec_sketches[phase][j].update(i,j,-1)
            #print(f"Added ({i},{j}) to sketch S_[{phase}](a^{j})", flush=True)
    # for k in range(n):
    #     vecs = []
    #     for i,j in in_graph.iter_edges():
    #         delta = 1
    #         if k == i:
    #             vecs.append((i,j,delta))
    #         elif k == j:
    #             vecs.append((i,j,-delta))
    #     print(f"Testin, len={len(vecs)}", flush=True)
    #     for phase in range(NUM_SKETCHES):
    #         if vec_sketches[phase] != None:
    #             vec_sketches[phase].append
    #         else:
    #             vec_sketches[phase] = [L0Sketch(n, seed=sketch_seeds[phase])]
    #         if vecs == []:
    #             continue
    #         for i,j,delta in vecs:
    #             print(f"Adding ({i},{j}) to sketch {k}", flush=True)
    #             vec_sketches[phase][k].update(i,j,delta)
    component = list(range(n))
    uniq_component = n
    
    for phase in range(NUM_SKETCHES):
        out_edges = [None] * n
        in_edges = [None] * n
        if uniq_component == 1:
            break
        for k in range(n-1,-1, -1):
            if component[k] == k:
                sample = vec_sketches[phase][k].sample()
                #print(f"Round {phase}: sampled {sample} from supernode at {k}")
                if sample == None:
                    continue
                else:
                    in_edges[k] = sample[0]
                    out_edges[k] = sample[1]
        #print(f"Round {phase}: out_edges = {out_edges}")
        for k in range(n-1,-1, -1):
            if out_edges[k] == None:
                continue
            else:
                x = component[k]
                y = component[out_edges[k]]
                repair_nodes = [k, out_edges[k]]
                while (x != component[x]):
                    repair_nodes.append(x)
                    x = component[x]
                while (y != component[y]):
                    repair_nodes.append(y)
                    y = component[y]
                new_comp = min(x,y)
                component[x] = new_comp
                component[y] = new_comp
                if x != y:
                    e = graph.edge(in_edges[k], out_edges[k])
                    if not e or not e.is_valid():
                        print(f"Why is this invalid: ({in_edges[k]},{out_edges[k]})?")

                    else:
                        graph.ep["s"][graph.edge(in_edges[k], out_edges[k])] = 1.0
                        
                    #print(f"Merging components {x} and {y} due to edge ({k},{out_edges[k]})")                    
                    uniq_component -= 1
                for node in repair_nodes:
                        component[node] = new_comp
                        #print(f"Node {node} is now in supernode {new_comp}")
        for k in range(n-1, -1, -1):
            if component[k] < k:
                x = component[k]
                repair_nodes = [k]
                while x != component[x]:
                    repair_nodes.append(x)
                    x = component[x]
                for node in repair_nodes:
                    component[node] = x
            #print(f"Round {phase}: component: {component}")
        for k in range(n-1,-1,-1):
            if component[k] < k:
                if phase < NUM_SKETCHES-1:
                    vec_sketches[phase+1][component[k]] += vec_sketches[phase+1][k]
                    #print(f"In round {phase+1}, supernode {component[k]} contains {k} as a subnode")
    print(f"Components: {component}")
    print(f"The algorithm found {uniq_component} different components")
    draw.graphviz_draw(graph, penwidth=graph.ep["s"], output="connected.png")        
    # print(f"(9,2) -> {sub1.sample()}")
    # print(f"(3,0) -> {sub2.sample()}")
    # print(f"(4,5) -> {sub3.sample()}")
    # print(f"(whole graph) -> {sub4.sample()}")


if __name__=='__main__':
    main()