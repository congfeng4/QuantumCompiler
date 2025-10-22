import itertools as it
import random as rd

from rustworkx import PyGraph
from typing import Iterator, Optional, TypeVar, Sequence
Self = TypeVar("Self")

from .graph_data import Edge, Node


class Graph:
    """
    A graph class, powered by rustworkx's PyGraph, with support for graph union
    and graph permutation.
    """
    __pygraph: PyGraph
    __name: str
    __i2n: dict[int, Node] # index-to-node map
    __n2i: dict[Node, int] # node-to-index map

    def __init__(self, pygraph: PyGraph, name: str = "graph"):

        if pygraph.has_parallel_edges():
            raise ValueError("graph contains parallel edges")
        if not all(isinstance(node, Node) for node in pygraph.nodes()):
            raise ValueError("graph nodes must be of type Node")

        self.__pygraph = pygraph
        self.__name = name
        self.__i2n = {index: pygraph[index] for index in pygraph.node_indices()}
        self.__n2i = {pygraph[index]: index for index in pygraph.node_indices()}

    @classmethod
    def from_edges(cls, edges: list[tuple[int, int]]) -> Self:
        """
        Instantiate a graph from the provided edge list. This method also ensures that
        nodes are indexed consecutively.

        Params:
        - `edges`: the list of edges to instantiate the graph from

        Returns:
        - a graph with provided edge list
        """
        # Re-index nodes and edges from edge list
        old_nodes = sorted(set(node for edge in edges for node in edge))
        new_nodes = [Node(v) for v in range(len(old_nodes))]
        new_edges = [
            Edge(Node(old_nodes.index(src)), Node(old_nodes.index(dst)))
            for src, dst in edges
        ]
        # Construct graph
        pygraph = PyGraph(multigraph = False)
        pygraph.add_nodes_from(new_nodes)
        n2i = {pygraph[index]: index for index in pygraph.node_indices()}
        pygraph.add_edges_from([(n2i[src], n2i[dst], Edge(src, dst)) for src, dst in new_edges])

        return cls(pygraph)

    def __eq__(self, other: Self) -> bool:
        return set(self.nodes) == set(other.nodes) and set(self.edges) == set(other.edges)

    def __getitem__(self, key: Union[int, Node]) -> Node | int:
        return self.__i2n[key] if isinstance(key, int) else self.__n2i[key]

    def __repr__(self) -> str:
        return "Graph(\n" + \
            f"  nodes: {self.nodes}\n" + \
            f"  edges: {self.edges}\n" + \
        ")"

    @property
    def name(self) -> str:
        return self.__name

    @property
    def nodes(self) -> list[Node]:
        return self.__pygraph.nodes()

    @property
    def num_nodes(self) -> int:
        return self.__pygraph.num_nodes()

    @property
    def num_edges(self) -> int:
        return self.__pygraph.num_edges()

    @property
    def edges(self) -> list[Edge]:
        return self.__pygraph.edges()

    def pygraph(self) -> PyGraph:
        return self.__pygraph

    def copy(self) -> Self:
        return Graph(self.__pygraph.copy())

    def add_node(self, node: Node) -> None:
        self.add_nodes([node])

    def add_nodes(self, nodes: Sequence[Node]) -> None:
        indices = self.__pygraph.add_nodes_from(nodes)
        for node, index in zip(nodes, indices):
            self.__i2n[index] = node
            self.__n2i[node] = index

    def remove_node(self, node: Node) -> None:
        self.remove_nodes([node])

    def remove_nodes(self, nodes: Sequence[Node]) -> None:
        self.__pygraph.remove_nodes_from([self[node] for node in nodes])
        for node in nodes:
            del self.__i2n[self[node]]
            del self.__n2i[node]

    def add_edge(self, edge: Edge) -> None:
        self.add_edges([edge])

    def add_edges(self, edges: Sequence[Edge]) -> None:
        self.__pygraph.add_edges_from([
            (self[src], self[dst], Edge(src, dst)) for src, dst in edges
        ])

    def remove_edge(self, edge: Edge) -> None:
        self.remove_edges([edge])

    def remove_edges(self, edges: Sequence[Edge]) -> None:
        self.__pygraph.remove_edges_from([
            (self[src], self[dst]) for src, dst in edges
        ])

    def has_node(self, node: Node) -> bool:
        return node in self.__n2i

    def has_edge(self, src: Node, dst: Node) -> bool:
        if not (self.has_node(src) and self.has_node(dst)):
            return False
        return self.__pygraph.has_edge(self[src], self[dst])

    def neighbours(self, node: Node) -> list[Node]:
        return [self[index] for index in self.__pygraph.neighbors(self[node])]

    def incident_edges(self, src: Node) -> list[Edge]:
        return [Edge(src, dst) for dst in self.neighbours(src)]

    def random_subgraph(self, num_edges: int) -> Self:
        """
        Generate a random edge-induced subgraph with the specified number of edges.

        Params:
        - `num_edges`: number of edges in the generated subgraph

        Returns:
        - random edge-induced subgraph of this graph
        """
        edges = [(self[src], self[dst]) for src, dst in rd.sample(self.edges, num_edges)]
        return Graph(self.__pygraph.edge_subgraph(edges))

    def random_nodes(self, num_nodes: int, include_all: bool = False) -> list[Node]:
        """
        Randomly pick nodes from this graph with replacement.

        Params:
        - `num_nodes`: number of nodes to sample
        - `include_all`: if this is True then the returned list is guaranteed to include
        every node in the graph at least once

        Returns:
        - list of randomly sampled nodes
        """
        if include_all and num_nodes < self.num_nodes:
            raise ValueError("cannot ensure every node is included with given num_nodes")

        nodes = []
        if include_all:
            nodes += rd.sample(self.nodes, self.num_nodes)
            num_nodes -= self.num_nodes
        nodes += rd.choices(self.nodes, k = num_nodes)

        return nodes

    def random_edges(self, num_edges: int, include_all: bool = True) -> list[Edge]:
        """
        Randomly pick edges from this graph with replacement.

        Params:
        - `n`: number of edges to sample
        - `include_all`: if this is True then the returned list is guaranteed to include
        every edge in the graph at least once

        Returns:
        - list of randomly sampled edges
        """
        if include_all and num_edges < self.num_edges:
            raise ValueError("cannot ensure every edge is included with given num_edges")

        edges = []
        if include_all:
            edges += rd.sample(self.edges, self.num_edges)
            num_edges -= self.num_edges
        edges += rd.choices(self.edges, k = num_edges)

        return edges

    def permute(self, src: Node, dst: Node, inplace: bool = False) -> Optional[Self]:
        """
        Permute nodes src and dst on this graph.

        Params:
        - `src`: the node to swap with dst
        - `dst`: the node to swap with src
        - `inplace`: if True then the permutation is done on this graph

        Returns:
        - permuted graph if `inplace` is False
        """
        if isinstance(src, int) or isinstance(dst, int):
            raise ValueError("src and dst should be of type Node")

        # Do not permute if neither src nor dst exists in graph
        if not (self.has_node(src) or self.has_node(dst)):
            return self.copy() if not inplace else None

        this = self
        if not inplace:
            this = self.copy()

        # Add external nodes to graph
        src_is_external = False
        if not this.has_node(src):
            this.add_node(src)
            src_is_external = True

        dst_is_external = False
        if not this.has_node(dst):
            this.add_node(dst)
            dst_is_external = True

        # Obtain neighbours of src and dst
        src_neighbors = set(this.neighbours(src)) - {dst}
        dst_neighbors = set(this.neighbours(dst)) - {src}

        # Disconnect u and v from their neighbours
        this.remove_edges([Edge(src, ngb) for ngb in src_neighbors])
        this.remove_edges([Edge(dst, ngb) for ngb in dst_neighbors])

        # Connect u to v's neighbours and v to u's neighbours
        this.add_edges([Edge(dst, ngb) for ngb in src_neighbors])
        this.add_edges([Edge(src, ngb) for ngb in dst_neighbors])

        # Remove its counterpart after permuting if a node was external
        if src_is_external:
            this.remove_node(src)
        if dst_is_external:
            this.remove_node(dst)

        if not inplace:
            return this

    def union(self, other: Self, inplace: bool = False) -> Optional[Self]:
        """
        Construct the union graph of this graph and the other graph.

        Params:
        - `other`: graph to union with this graph
        - `inplace`: if True then the union is performed direcly on this graph

        Returns:
        - union graph if `inplace` is False
        """
        this = self
        if not inplace:
            this = this.copy()

        # Add union of nodes and edges
        this.add_nodes(list(set(other.nodes) - set(this.nodes)))
        this.add_edges(other.edges)

        if not inplace:
            return this

    def __or__(self, other: Self) -> Self:
        return self.union(other)


if __name__ == "__main__":

    graph1 = Graph.from_edges(
        rd.sample(list(it.combinations(range(10), 2)), k = 10)
    )
    # Test permute
    src, dst = Node(0), Node(5)
    swapped_graph = graph1.permute(src, dst)
    if graph1.neighbours(src) != graph1.neighbours(dst):
        assert graph1 != swapped_graph
    assert graph1.nodes == swapped_graph.nodes
    assert len(graph1.edges) == len(swapped_graph.edges)
    for i in range(len(graph1.edges)):
        src, dst = graph1.edges[i]
        if len({src.__val, dst.__val} & {0, 5}) in (0, 2):
            continue
        if src.__val in (0, 5):
            assert swapped_graph.has_edge(Node(({0, 5} - {src.__val}).pop()), dst)
        else:
            assert swapped_graph.has_edge(src, Node(({0, 5} - {dst.__val}).pop()))

    # Test union
    graph2 = Graph.from_edges([(0, 1), (1, 3), (2, 4), (3, 5)])
    assert (union_graph := graph1.union(graph2)) == (graph1 | graph2)
    assert set(union_graph.nodes) == set(graph1.nodes) | set(graph2.nodes)
    assert set(union_graph.edges) == set(graph1.edges) | set(graph2.edges)
