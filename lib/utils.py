from enum import Enum
from rustworkx import is_subgraph_isomorphic
from typing import Iterable, Optional

from .graph import Graph
from .graph_data import Edge
from .permutation import Permutation

VF2_CALL_LIMIT = 10000


# === QUEKNO parameters ===

class OptType(Enum):
    OPT1 = 1
    OPT2 = 2
    DEPTH = None

class SubgraphSize(Enum):
    TOKYO = 5
    SMALL = 8
    LARGE = 16

class QBGRatio(Enum):
    TFL = 1.5
    QSE = 2.55


def is_disjoint(edges: Iterable[Optional[Edge]]) -> bool:
    """
    Check whether the given collection of edges is disjoint.

    Params:
    - `edges`: an iterable collection of edges

    Returns:
    - True if `edges` is (pairwise) disjoint.
    """
    nodes = [node for edge in edges for node in edge if not edge.is_null]
    return len(nodes) == len(set(nodes)) # disjoint if no duplicates


def is_strong_glink(
    archgraph: Graph,
    prev_subgraph: Graph,
    next_subgraph: Graph,
    perm: Permutation
) -> bool:
    """
    Determine whether the given glink is strong.

    Params:
    - `archgraph`: architecture graph
    - `prev_subgraph`: previous subgraph in the glink
    - `next_subgraph`: next subgraph in the glink
    - `perms`: permutation inducing the glink

    Returns:
    - True if the induced glink is strong
    """
    # Construct permuted graph
    perm_graph = next_subgraph.copy()
    for src, dst in perm.items():
        perm_graph.permute(src, dst, inplace = True)

    # If no changes have been made then the glink cannot be strong
    if perm_graph == next_subgraph:
        return False

    # Glink is strong if its union graph is not isomorphic to archgraph
    return is_subgraph_isomorphic(
        archgraph.pygraph(),
        prev_subgraph.union(perm_graph).pygraph(),
        induced = False,
        call_limit = VF2_CALL_LIMIT
    )
