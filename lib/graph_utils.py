from rustworkx import generators

from .graph import Graph


def graph_from_name(name: str, **kwargs) -> Graph:

    match name.lower():
        case "grid":
            return Grid(kwargs["rows"], kwargs["cols"])
        case "line":
            return Line(kwargs["num_nodes"])
        case "ring":
            return Ring(kwargs["num_nodes"])
        case "star":
            return Star(kwargs["num_nodes"])
        case "tokyo":
            return Tokyo()
        case "rochester":
            return Rochester()
        case "sycamore54":
            return Sycamore54()
        case "sycamore":
            return Sycamore()
        case _:
            raise ValueError(f"unknown graph name '{name}'")


class Grid(Graph):
    """
    Grid graph.
    """
    def __init__(self, rows: int, cols: int):
        edges = generators.grid_graph(rows, cols).edge_list()
        grid = Graph.from_edges(edges).pygraph()
        super().__init__(grid, f"grid({rows}, {cols})")


class Line(Graph):
    """
    Path graph (not to be confused with line graph).
    """
    def __init__(self, num_nodes: int):
        edges = generators.path_graph(num_nodes).edge_list()
        line = Graph.from_edges(edges).pygraph()
        super().__init__(line, f"line({num_nodes})")


class Ring(Graph):
    """
    Cycle graph.
    """
    def __init__(self, num_nodes: int):
        edges = generators.cycle_graph(num_nodes).edge_list()
        ring = Graph.from_edges(edges).pygraph()
        super().__init__(ring, f"ring({num_nodes})")


class Star(Graph):
    """
    Star graph.
    """
    def __init__(self, num_nodes: int):
        edges = generators.star_graph(num_nodes).edge_list()
        star = Graph.from_edges(edges).pygraph()
        super().__init__(star, f"star({num_nodes})")


class Tokyo(Graph):
    """
    IBM Q Tokyo (20 qubits).
    """
    def __init__(self):

        edges = sorted([
            ( 0,  1), ( 1,  2), ( 2,  3), ( 3,  4),
            ( 0,  5), ( 1,  6), ( 1,  7), ( 2,  6), ( 2,  7), ( 3,  8), ( 3,  9), ( 4,  8), ( 4,  9),
            ( 5,  6), ( 6,  7), ( 7,  8), ( 8,  9),
            ( 5, 10), ( 5, 11), ( 6, 10), ( 6, 11), ( 7, 12), ( 7, 13), ( 8, 12), ( 8, 13), ( 9, 14),
            (10, 11), (11, 12), (12, 13), (13, 14),
            (10, 15), (11, 16), (11, 17), (12, 16), (12, 17), (13, 18), (13, 19), (14, 18), (14, 19),
            (15, 16), (16, 17), (17, 18), (18, 19)
        ])
        tokyo = Graph.from_edges(edges).pygraph()
        super().__init__(tokyo, "tokyo")


class Rochester(Graph):
    """
    IBM Q Rochester (53 qubits).
    """
    def __init__(self):

        edges = sorted([
            ( 0,  1), ( 1,  2), ( 2,  3), ( 3,  4),
            ( 0,  5), ( 4,  6),
            ( 5,  9), ( 6, 13),
            ( 7,  8), ( 8,  9), ( 9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
            ( 7, 16), (11, 17), (15, 18),
            (16, 19), (17, 23), (18, 27),
            (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
            (21, 28), (25, 29),
            (28, 32), (29, 36),
            (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38),
            (30, 39), (34, 40), (38, 41),
            (39, 42), (40, 46), (41, 50),
            (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50),
            (44, 51), (48, 52)
        ])
        rochester = Graph.from_edges(edges).pygraph()
        super().__init__(rochester, "rochester")


class Sycamore54(Graph):
    """
    Google Sycamore (54 qubits).
    """
    def __init__(self):

        edges = sorted([
            ( 0,  6), ( 1,  6), ( 1,  7), ( 2,  7), ( 2,  8), ( 3,  8), ( 3,  9), ( 4,  9), ( 4, 10), ( 5, 10), ( 5, 11),
            ( 6, 12), ( 6, 13), ( 7, 13), ( 7, 14), ( 8, 14), ( 8, 15), ( 9, 15), ( 9, 16), (10, 16), (10, 17), (11, 17),
            (12, 18), (13, 18), (13, 19), (14, 19), (14, 20), (15, 20), (15, 21), (16, 21), (16, 22), (17, 22), (17, 23),
            (18, 24), (18, 25), (19, 25), (19, 26), (20, 26), (20, 27), (21, 27), (21, 28), (22, 28), (22, 29), (23, 29),
            (24, 30), (25, 30), (25, 31), (26, 31), (26, 32), (27, 32), (27, 33), (28, 33), (28, 34), (29, 34), (29, 35),
            (30, 36), (30, 37), (31, 37), (31, 38), (32, 38), (32, 39), (33, 39), (33, 40), (34, 40), (34, 41), (35, 41),
            (36, 42), (37, 42), (37, 43), (37, 43), (38, 44), (39, 44), (39, 45), (40, 45), (40, 46), (41, 46), (41, 47),
            (42, 48), (42, 49), (43, 49), (43, 50), (44, 50), (44, 51), (45, 51), (45, 52), (46, 52), (46, 53), (47, 53)
        ])
        sycamore54 = Graph.from_edges(edges).pygraph()
        super().__init__(sycamore54, "sycamore54")


class Sycamore(Graph):
    """
    Google Sycamore (53 qubits).
    """
    def __init__(self):

        sycamore54 = Sycamore54().pygraph()
        sycamore54.remove_node(3) # remove bad qubit
        edges = sycamore54.edges()
        sycamore53 = Graph.from_edges(edges).pygraph()
        super().__init__(sycamore53, "sycamore")


if __name__ == "__main__":

    tokyo = Tokyo()
    rochester = Rochester()
    sycamore = Sycamore()

    # Test inheritance
    print(f"{isinstance(tokyo, Tokyo) = }")
    print(f"{isinstance(tokyo, Rochester) = }")
    print(f"{isinstance(tokyo, Sycamore) = }")
    print(f"{isinstance(tokyo, Graph) = }")
