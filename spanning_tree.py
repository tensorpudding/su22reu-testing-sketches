from tkinter import N
from l_zero_graph import L0Sketch
import numpy as np
import graph_tool as gt

def spanning_tree(graph_stream, n, seed=None, weighted=False):
    """
    Accepts a stream of updates to edges (u,v,delta), where positive delta adds a weighted edge (u,v), and negative delta removes.
    n represents the number of nodes in the graph
    Returns a graph that is a spanning tree of the input using a modified form of Boruvka's algorithm for spanning trees
    This algorithm uses log(n) L0 sketches for each vertex to sample edges incident to each vertex
    The L0 sketches each take O(log^2(n)) space and thus the whole structure is O(n*log^3(n))
    The supernodes are then collapsed into one component, and their sketches are added linearly
    Edges sampled from the sum of sketches will be those incident to edges from the supernode composed of those vertices
    Boruvka's algorithm yields a spanning forest after the log(n) rounds completed.
    """
    component = list(range(n))
    uniq_components = n
    graph = gt.Graph(directed=False)
    graph.add_vertex(n)
    if not seed:
        rng = np.default_rng()
    else:
        rng = np.default_rng(seed)
    NUM_SKETCHES=int(2*np.log(n))
    sketch_seeds = rng.integers(2**32, size=NUM_SKETCHES)

    # vec_sketches represents a list of lists of sketches
    # vec_sketches[i][j] is the ith sketch of the jth index
    # We need O(log(n)) sketches for each vertex to guarantee independence
    # The ith sketch of each vertex uses the same structure, allowing them to be added
    vec_sketches = []
    for i in range(NUM_SKETCHES):
        vec_sketches.append([])
        for j in range(n):
            vec_sketches[i].append(L0Sketch(n, seed=sketch_seeds[i]))
    for i, j, delta in graph_stream:
        if j == i:
            continue
        if j > i:
            temp = i
            i = j
            j = temp
        for k in range(NUM_SKETCHES):
            vec_sketches[k][i].update(i,j,delta)
            vec_sketches[k][j].update(i,j,-delta)
    
    # Main algorithm

    for i in range(NUM_SKETCHES):
        if uniq_components == 1:
            break
        # Initialize the arrays representing sampled edges
        out_edges = [None] * n
        in_edges = [None] * n
        # First step: sampling edges for each component
        # Each component is represented by the sketch at the vertex of lowest index
        # This will be updated to be the sum of all sketches at that level for that component
        for k in range(n-1,-1, -1):
            if component[k] == k:
                sample = vec_sketches[i][k].sample()
                #print(f"Round {i}: sampled {sample} from supernode at {k}")
                # Sometimes this fails, either due to bad luck or due to the component being maximal
                if sample == None:
                    in_edges[k] = None
                    out_edges[k] = None
                else:
                    in_edges[k] = sample[0]
                    out_edges[k] = sample[1]
        #print(f"Round {i}: out_edges = {out_edges}")
        # Second step: we update components to merge the two supernodes
        for k in range(n-1,-1, -1):
            if out_edges[k] == None:
                continue
            # Here we proceed to find the representative of the new connected component containing both of the previous connected components
            # An index k represents a component only if component[k] = k
            # So we must repair all nodes in the chain to connect two components
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
                # If the two components are indeed different, let's add the edge we added to the spanning tree
                # They might be the same if an edge in the same phase has already connected these two components
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
        # Third step: now we go back and fix nodes that were missed in the first pass, to update their components
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
        # Fourth step: the i+1 phase sketches of components are updated such that each one is the sum of the sketches of their constituent vertices
        # We thus find all nodes that are not the component representative and add their i+1st sketch to the i+1st sketch of the representative
        # This only need be done if we have not finished log(n) rounds, however
        for k in range(n-1,-1,-1):
            if component[k] < k:
                if i < NUM_SKETCHES-1:
                    vec_sketches[i+1][component[k]] += vec_sketches[i+1][k]
                    #print(f"In round {phase+1}, supernode {component[k]} contains {k} as a subnode")
        return graph