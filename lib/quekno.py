import math
import random as rd
import time
import warnings

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitError
from typing import Iterator

from .glink import GlinkChain
from .graph import Graph
from .graph_data import Edge
from .permutation import Permutation
from .utils import *
from config import *


class QUEKNO:
    __opt_type: OptType
    __target_cost: int
    __archgraph: Graph
    __subgraph_size: SubgraphSize
    __qbg_ratio: QBGRatio

    def __init__(
        self,
        opt_type: OptType,
        target_cost: int,
        archgraph: Graph,
        subgraph_size: SubgraphSize,
        qbg_ratio: QBGRatio
    ):
        self.__opt_type = opt_type
        self.__target_cost = target_cost
        self.__archgraph = archgraph
        self.__subgraph_size = subgraph_size
        self.__qbg_ratio = qbg_ratio
    
    @property
    def opt_type(self) -> OptType:
        return self.__opt_type
    
    @property
    def target_cost(self) -> int:
        return self.__target_cost
    
    @property
    def archgraph(self) -> Graph:
        return self.__archgraph
    
    @property
    def subgraph_size(self) -> int:
        return self.__subgraph_size.value
    
    @property
    def qbg_ratio(self) -> float:
        return self.__qbg_ratio.value

    def random_subgraph(self) -> Graph:
        """
        Generate a random subgraph with average number of edges as per `self.subgraph_size`.
        """
        num_edges = math.ceil(rd.gauss(self.subgraph_size, SUBGRAPH_SIZE_STD))
        num_edges = max(num_edges, 1)
        num_edges = min(num_edges, self.archgraph.num_edges)
        return self.archgraph.random_subgraph(num_edges)

    def __consecutive_permutations(self, num_swaps: int) -> Iterator[Permutation]:

        if num_swaps not in (1, 2):
            raise ValueError("num_swaps needs to be either 1 or 2")
        
        edges1 = self.archgraph.edges
        rd.shuffle(edges1)
        
        for src1, dst1 in edges1:
            
            # For opt1, we can return straight away
            if num_swaps == 1:
                yield Permutation(Edge(src1, dst1), type = "swap")
                continue
            
            # Generate consecutive swaps
            src1_inc_edges = set(self.archgraph.incident_edges(src1))
            dst1_inc_edges = set(self.archgraph.incident_edges(dst1))
            edges2 = list((src1_inc_edges | dst1_inc_edges) - {Edge(src1, dst1)})
            rd.shuffle(edges2)
            
            # For opt2, we randomly choose to include a second consecutive swap
            for src2, dst2 in edges2:
                num_swaps = rd.choices((1, 2), (.5 - CONSEC_SWAPS_BIAS, .5 + CONSEC_SWAPS_BIAS))[0]

                # Select one swap
                if num_swaps == 1:
                    yield Permutation(Edge(src1, dst1), type = "swap")
                    continue
                
                # Select two swaps
                if src1 == src2:
                    src1, dst1 = dst1, src1
                elif src1 == dst2:
                    src1, dst1 = dst1, src1
                    src2, dst2 = dst2, src2
                elif dst1 == dst2:
                    src2, dst2 = dst2, src2
                yield Permutation(Edge(src1, dst1), Edge(src2, dst2), type = "swap")

    def __parallel_permutations(self) -> Iterator[Permutation]:

        while True:
        
            cand_edges = self.archgraph.edges # candidate edges
            parallel_edges = {rd.choice(cand_edges)} # selected parallel edges
            cand_edges += [Edge.null()] # add a null edge for random early break

            while cand_edges:
                cand_edges = [edge for edge in cand_edges if is_disjoint({edge} | parallel_edges)]
                if (edge := rd.choice(cand_edges)).is_null: # stop selecting more edges
                    break
                parallel_edges.add(edge)
            
            yield Permutation(*parallel_edges, type = "swap")

    def permutations(self, cost: int):
        """
        Return a generator of glink-inducing permutations.

        Params:
        - `cost`: current cost

        Yields:
        - a permutation inducing the glink
        """
        if self.opt_type != OptType.DEPTH:
            num_swaps = min(self.opt_type.value, self.target_cost - cost)
            perms = self.__consecutive_permutations(num_swaps)
        else:
            perms = self.__parallel_permutations()
        return perms

    def next_glink(self, glink_chain: GlinkChain, cost: int) -> tuple[Graph, Permutation]:
        """
        Find the next (strong) glink in the given glink chain.

        Params:
        - `glink_chain`: the current glink chain
        - `cost`: current cost

        Returns:
        - the subgraph and permutation inducing the next glink
        """
        prev_subgraph = glink_chain.tail.graph
        done = False

        while not done:

            # Randomly generate next subgraph
            next_subgraph = self.random_subgraph()

            # Generate glink-inducing permutations
            perms = self.permutations(cost)
            next_perm = None

            # Try each permutation until a non-isomorphic union graph is induced
            for _ in range(GLINK_SEARCH_PATIENCE):
                try:
                    next_perm = next(perms)
                except StopIteration:
                    break # if all perms have been exhausted, regenerate dst_subgraph
                if (done := is_strong_glink(self.archgraph, prev_subgraph, next_subgraph, next_perm)):
                    break
        
        return next_subgraph, next_perm
    
    def build_glink_chain(self) -> tuple[GlinkChain, int]:
        """
        Construct a (strong) glink chain with respect to `self.graph`.

        Returns:
        - the constructed glink chain
        - the induced transformation cost
        """
        chain = GlinkChain()
        cost = 0

        # Intialise first permutation and subgraph
        chain.append(
            graph = self.random_subgraph(),
            perm = Permutation.random(self.archgraph.nodes)
        )
        # No need to add more glinks if target cost is 0
        if self.target_cost == 0:
            return chain, cost
        
        while cost < self.target_cost:
            next_subgraph, next_perm = self.next_glink(chain, cost)
            chain.append(next_subgraph, next_perm)
            cost += len(next_perm) if self.opt_type != OptType.DEPTH else 1
        
        return chain, cost

    def build_circuit(
        self,
        glink_chain: GlinkChain,
        add_barriers: bool = False
    ) -> QuantumCircuit:
        """
        Construct a QUEKNO circuit from the given glink chain.

        Params:
        - `glink_chain`: the glink chain to construct the circuit from
        - `add_barriers`: if True then a barrier is added between every glink
        in the circuit
        """
        circuit = QuantumCircuit(self.archgraph.num_nodes)
        original = self.archgraph.nodes # current permutation of AG nodes
        
        for glink in glink_chain.glinks():

            # Apply the permutation inducing glink
            permuted = glink.perm.apply(original)
            if permuted == original:
                raise ValueError(f"identity permutation: {glink.perm}")

            # Retrieve edges that will be affected by the permutation
            front_gates = []
            for edge in self.archgraph.edges:
                original_edge = {original.index(node) for node in edge}
                permuted_edge = {permuted.index(node) for node in edge}
                if original_edge == permuted_edge:
                    continue
                front_gates.append(edge)

            # Sample random edges in subgraph for 2-qubit gates
            num_back_2qbgs = math.ceil(glink.graph.num_edges * (1 + RAND_EDGES_VAR * rd.randint(1, 4)))
            back_2qbgs = glink.graph.random_edges(num_back_2qbgs, include_all = True)

            # Sample random nodes in subgraph for 1-qubit gates
            num_back_1qbgs = math.ceil((len(front_gates) + len(back_2qbgs)) * self.qbg_ratio)
            back_1qbgs = glink.graph.random_nodes(num_back_1qbgs, include_all = False)

            # Generate gate list
            back_gates = back_2qbgs + back_1qbgs
            rd.shuffle(back_gates)
            gate_list = front_gates + back_gates

            # Add gates to circuit
            for gate in gate_list:
                gate_type = TWO_QUBIT_GATE if isinstance(gate, Edge) else ONE_QUBIT_GATE
                gate = gate if isinstance(gate, Edge) else [gate]
                circuit.append(gate_type, tuple(map(permuted.index, gate)))
            if add_barriers and glink.next is not None:
                circuit.barrier()

            # Update current permutation
            original = permuted

        return circuit
    
    def route(
        self,
        circuit: QuantumCircuit,
        glink_chain: GlinkChain,
        pred_cost: int,
        verbose: bool = True
    ) -> QuantumCircuit:
        """
        Perform routing on the given circuit to become executable on `self.archgraph`.

        Params:
        - `circuit`: circuit to be routed
        - `glink_chain`: glink chain inducing the circuit
        - `pred_cost`: predicted (expected) transformation cost
        - `verbose`: if True then each layout update is displayed

        Returns:
        - the transformed circuit
        """
        glink = glink_chain.head
        layout = glink.perm.apply(self.archgraph.nodes) # current layout
        routed_circuit = circuit.copy_empty_like()

        if verbose:
            print(f"layout: {Permutation().oneline(self.archgraph.nodes)} [it 0]")
            print(f"layout: {glink.perm.oneline(self.archgraph.nodes)} [it 1]")

        i = 0
        true_cost = 0
        while i < len(circuit.data):
            gate = circuit.data[i]
            
            if gate.operation.name == "barrier":
                routed_circuit.barrier() # barriers are ignored

            elif gate.operation == ONE_QUBIT_GATE:
                routed_circuit.append(ONE_QUBIT_GATE, gate.qubits) # add one-qubit gate

            elif gate.operation == TWO_QUBIT_GATE:
                permuted_qubits = [layout[qubit._index] for qubit in gate.qubits]
                if not self.archgraph.has_edge(*permuted_qubits):
                    
                    glink = glink.next
                    if glink is None:
                        raise CircuitError("too few glinks")
                    if verbose:
                        print(f"layout: {glink.perm.oneline(layout)} [it {i + 1}]")
                    
                    for src, dst in glink.perm.items():
                        routed_circuit.swap(layout.index(src), layout.index(dst)) # add swaps
                    layout = glink.perm.apply(layout) # update layout
                    true_cost += len(glink.perm) if self.opt_type != OptType.DEPTH else 1 # update cost
                    continue
                
                routed_circuit.append(TWO_QUBIT_GATE, gate.qubits) # add two-qubit gate

            else:
                raise CircuitError(f"unknown gate '{gate.operation.name}'")
            
            i += 1
        
        if glink.next is not None:
            raise CircuitError("too many glinks")
        
        if pred_cost != true_cost:
            raise CircuitError(f"{pred_cost = }, {true_cost = })")
        
        swap_cost = routed_circuit.size() - circuit.size()
        if self.opt_type != OptType.DEPTH and swap_cost != true_cost:
            warnings.warn(f"{swap_cost = }, {true_cost = }")
        
        depth_cost = routed_circuit.depth() - circuit.depth()
        if self.opt_type == OptType.DEPTH and depth_cost != true_cost:
            warnings.warn(f"{depth_cost = }, {true_cost = }")
            
        return routed_circuit
    
    def run(
        self,
        add_barriers: bool = False,
        verbose: bool = True
    ) -> tuple[QuantumCircuit, QuantumCircuit, dict]:
        """
        Construct a QUEKNO circuit and save results.

        Params:
        - `add_barriers`: if True then a barrier is added between every glink
        in the circuit
        - `verbose`: if True then verification will be verbose
        
        Returns:
        - QUEKNO circuit
        - routed QUEKNO circuit
        - dict containing results
        """
        # Construct
        t0 = time.time()
        glink_chain, cost = self.build_glink_chain()
        circuit = self.build_circuit(glink_chain, add_barriers)
        t1 = time.time()
        
        # Route
        routed_circuit = self.route(circuit, glink_chain, cost, verbose = verbose)

        # Record results
        gate_counts = circuit.count_ops()
        subgraph_sizes = [glink.graph.num_edges for glink in glink_chain.glinks()]
        results = {
            # parameters
            "opt_type": f"opt{self.opt_type.value}" if self.opt_type != OptType.DEPTH else "depth",
            "cost": cost,
            "archgraph": self.archgraph.name,
            "subgraph_size": sum(subgraph_sizes) / len(subgraph_sizes),
            "qbg_ratio": gate_counts[ONE_QUBIT_GATE.name] / gate_counts[TWO_QUBIT_GATE.name],
            # generated circuit
            "gate_size": circuit.size(),
            "depth": circuit.depth(),
            "gate_cost": routed_circuit.decompose("swap").size() - circuit.size(),
            "depth_cost": routed_circuit.decompose("swap").depth() - circuit.depth(),
            # permutations
            "init_map": glink_chain.head.perm.oneline(highlight = False),
            "swaps": [glink.perm.items() for glink in glink_chain.glinks() if glink != glink_chain.head],
            # build time
            "build_time": t1 - t0
        }
        return circuit, routed_circuit, results


if __name__ == "__main__":

    from lib.graph_utils import Tokyo

    builder = QUEKNO(
        opt_type = OptType.OPT2,
        target_cost = 3,
        archgraph = Tokyo(),
        subgraph_size = SubgraphSize.SMALL,
        qbg_ratio = QBGRatio.TFL
    )
    circuit, routed_circuit, results = builder.run(add_barriers = True)
    print("=== RESULTS ===")
    for key, val in results.items():
        if isinstance(val, float):
            val = f"{val:.3f}"
        print(f"{key}: {val}")
