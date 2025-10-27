import random
from turtle import st
import networkx as nx
import numpy as np
from qiskit.transpiler import CouplingMap
import math
from contrib.common import IBMQHardwareArchitecture


def closest_factors(n):
    if n < 1:
        return "请输入正整数"

    # 从平方根开始向下查找
    sqrt_n = int(math.sqrt(n))

    # 找到最大的能整除n的数
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return (i, n // i)

    # 对于质数，返回1和自身
    return (1, n)


def generate_graph_by_model(graph_model: str, n: int, verbose=False):
    # assert graph_model in 'line star grid random'.split(), f'Bad model {graph_model}'
    def generate():
        if graph_model in ('line', 'ring', 'star'):
            # return nx.path_graph(n=n)
            return IBMQHardwareArchitecture(graph_model, num_nodes=n)
        # if graph_model == 'star':
        #     return nx.star_graph(n=n)
        if graph_model == 'grid':
            # g = nx.grid_2d_graph(m=m_or_d, n=n)
            # node_mapping = {node: idx for idx, node in enumerate(g.nodes)}
            # return nx.relabel_nodes(g, node_mapping)
            rows, cols = closest_factors(n)
            return IBMQHardwareArchitecture(graph_model, rows=rows, cols=cols)

        if graph_model == 'random':
            return generate_jellyfish_network(N=n, d=4)

        raise ValueError(f'Bad graph model: {graph_model}')

    g = generate()
    if verbose:
        print(f'Generate {graph_model} {n}')
        nx.draw(g)
    return g


def generate_jellyfish_network(N, d, iterations=10):
    """
    Generate a Jellyfish network as a NetworkX graph.

    Args:
        N (int): Number of nodes
        d (int): Fixed degree per node (must be even)
        iterations (int): Number of edge swaps (scaled by total edges)

    Returns:
        nx.Graph: Jellyfish network graph
    """
    if d % 2 != 0:
        raise ValueError("Degree must be even for Jellyfish networks")
    if N <= d:
        raise ValueError("Number of nodes must exceed degree")

    # Initialize graph
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # Create initial circular connections to ensure basic connectivity
    for i in range(N):
        for j in range(1, d // 2 + 1):
            neighbor = (i + j) % N
            if not G.has_edge(i, neighbor):
                G.add_edge(i, neighbor)

    # Calculate total edges and swap iterations
    total_edges = (N * d) // 2
    swap_iterations = iterations * total_edges

    # Perform random edge swaps to randomize the structure
    for _ in range(swap_iterations):
        # Find two valid edges for swapping
        while True:
            # Get first random edge (u1, v1)
            u1 = random.randint(0, N - 1)
            if G.degree(u1) < 1:
                continue
            v1 = random.choice(list(G.neighbors(u1)))
            if u1 > v1:
                u1, v1 = v1, u1

            # Get second random edge (u2, v2) that doesn't share nodes with first edge
            u2 = random.randint(0, N - 1)
            if u2 == u1 or u2 == v1 or G.degree(u2) < 1:
                continue
            v2 = random.choice(list(G.neighbors(u2)))
            if v2 == u1 or v2 == v1 or u2 == v2 or u2 > v2:
                continue

            # Ensure no existing edges between the cross nodes
            if not G.has_edge(u1, u2) and not G.has_edge(v1, v2):
                break

        # Perform edge swap: (u1-v1, u2-v2) → (u1-u2, v1-v2)
        G.remove_edge(u1, v1)
        G.remove_edge(u2, v2)
        G.add_edge(u1, u2)
        G.add_edge(v1, v2)

    return G


def create_coupling_graph(g: nx.Graph, check=False):
    coupling_map = CouplingMap(list(g.edges))
    if check:
        assert g.number_of_edges() == coupling_map.graph.num_edges()
        assert g.number_of_nodes() == coupling_map.graph.num_nodes()
        assert np.all(coupling_map.distance_matrix == nx.floyd_warshall_numpy(g))
        print(f'Check ok, |N| {coupling_map.graph.num_nodes()} |E| {coupling_map.graph.num_edges()}')
    return coupling_map


def generate_graph_for_num_qubits(graph_model: str, num_qubits: int, return_coupling_map=True):
    assert num_qubits > 0, num_qubits

    def gen():
        if isinstance(graph_model, str):
            return generate_graph_by_model(graph_model, n=num_qubits)

        if isinstance(graph_model, nx.Graph):
            assert graph_model.number_of_nodes() == num_qubits
            return graph_model

        raise TypeError(graph_model)

    g = gen()
    g_name = graph_model if isinstance(graph_model, str) else g.name

    if return_coupling_map:
        return create_coupling_graph(g), g_name

    return g, g_name
