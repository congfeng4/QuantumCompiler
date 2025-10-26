import subprocess
from copy import deepcopy
from typing import Union, List
import networkx as nx
from pathlib import Path
from qiskit import transpile
from qiskit import QuantumCircuit
from dataclasses import dataclass
import sys
from enum import Enum
import tempfile

from qiskit.transpiler import PassManager

from contrib.initial_mapping import InitialMappingStrategy
from contrib.random_graphs import generate_graph_for_num_qubits, create_coupling_graph
from contrib.action_space import OPT_PASSES
from contrib.common import qknob_metrics, get_total_ops, get_cnot_num, get_gate_set
from contrib.verify_circuit import verify_circuit_equivalent
from hamap import IBMQHardwareArchitecture

SUPPORTED_GRAPH_MODEL = ('line', 'star', 'grid', 'random')
SUPPORTED_OPT_LEVEL = tuple(range(3))


class OptOrder(Enum):
    BEFORE_ROUTING = 'before'
    AFTER_ROUTING = 'after'


class OptMethod(Enum):
    NONE = 'none'
    PASSES = 'passes'
    QUARL = 'quarl'
    QUARTZ = 'quartz'
    QISKIT_LV1 = 'qiskit:1'
    QISKIT_LV2 = 'qiskit:2'


class LayoutMethod(Enum):
    TRIVIAL = 'trivial'
    SABRE = 'sabre'

    def to_initial_mapping_strategy(self):
        if self == self.TRIVIAL:
            return InitialMappingStrategy.IDENTITY
        if self == self.SABRE:
            return InitialMappingStrategy.SABRE
        raise ValueError(self)


class RoutingMethod(Enum):
    SABRE = 'sabre'
    BASIC = 'basic'
    HA = 'ha'


SUPPORTED_LAYOUT_METHOD = (LayoutMethod.SABRE, LayoutMethod.TRIVIAL)  # dense
SUPPORTED_ROUTING_METHOD = (RoutingMethod.SABRE, RoutingMethod.HA, RoutingMethod.BASIC)  # lookahead
SUPPORTED_OPT_ORDER = (OptOrder.AFTER_ROUTING, OptOrder.BEFORE_ROUTING)
# SUPPORTED_OPT_METHOD = (OptMethod.QUARL, Op)

@dataclass
class CircuitStats:
    """
    Obtain statistics of circuits more conveniently
    """
    name: str
    num_qubits: int
    depth: int
    num_ops: int
    num_cx: int
    gate_set: List[str]
    qc: QuantumCircuit

    def __init__(self, qc: QuantumCircuit, name: str):
        self.name = name
        self.depth = qc.depth()
        self.num_ops = get_total_ops(qc)
        self.num_qubits = qc.num_qubits
        self.num_cx = get_cnot_num(qc)
        self.gate_set = get_gate_set(qc)
        self.qc = qc


def route_circuit(qc: QuantumCircuit, coupling_map,
                  routing_method: RoutingMethod, layout_method: LayoutMethod,
                  gate_set, optimization_level):

    if routing_method == RoutingMethod.HA:
        from hamap.mapping import ha_mapping
        from contrib.initial_mapping import get_initial_mapping

        initial_mapping = get_initial_mapping(qc, coupling_map, layout_method.to_initial_mapping_strategy())
        qc_output, _ = ha_mapping(qc, initial_mapping=initial_mapping, hardware=coupling_map)
        return qc_output

    if routing_method in (RoutingMethod.SABRE, RoutingMethod.BASIC):
        if isinstance(coupling_map, nx.Graph):
            coupling_map = create_coupling_graph(coupling_map)

        qc_output = transpile(qc,
                              basis_gates=gate_set,
                              coupling_map=coupling_map,
                              optimization_level=optimization_level,  # 0 for pure routing without optimization.
                              layout_method=layout_method.value,
                              routing_method=routing_method.value,
                              )
        return qc_output

    raise ValueError(routing_method)


def optimize_circuit(qc: QuantumCircuit, opt_method: OptMethod, opt_params: dict = None, verbose=False):
    if opt_method == OptMethod.NONE:
        return qc
    if opt_method == OptMethod.PASSES:
        return qiskit_pass_optimize(qc, OPT_PASSES)
    if opt_method == OptMethod.QUARL:
        return quarl_optimize(qc, gate_set=opt_params['gate_set'],
                              ecc_file=opt_params['ecc_file'],
                              quarl_dir=opt_params['quarl_dir'],
                              verbose=verbose,
                              )
    if opt_method == OptMethod.QUARTZ:
        return quartz_optimize(qc, gate_set=opt_params['gate_set'],
                               ecc_file=opt_params['ecc_file'],
                               verbose=verbose)
    if opt_method in (OptMethod.QISKIT_LV1, OptMethod.QISKIT_LV2):   # Router should optimize it.
        return qc

    raise ValueError(opt_method)


def qiskit_pass_optimize(qc: QuantumCircuit, opt_passes):
    pass_mngr = PassManager(opt_passes)
    return pass_mngr.run(deepcopy(qc))


def quarl_optimize(qc: QuantumCircuit,
                   gate_set,
                   ecc_file: Path,
                   quarl_dir: Path,
                   max_iterations=50,
                   verbose=True) -> QuantumCircuit:
    assert ecc_file.exists(), ecc_file
    assert quarl_dir.exists(), quarl_dir

    from qiskit.qasm2 import dumps

    gate_set_arg = '[' + ",".join(gate_set) + ']'

    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        input_qasm = tempdir / 'input.qasm'
        input_qasm.write_text(dumps(qc), encoding='utf8')
        output_qasm_file = tempdir / 'output.qasm'

        cmd = (f'{sys.executable} ppo.py '
               f'c.input_graphs.0.name={output_qasm_file.stem} '
               f'c.input_graphs.0.path={input_qasm} '
               f'c.ecc_file={ecc_file} '
               f'c.gate_set={gate_set_arg} '
               f'c.max_iterations={max_iterations} '
               f'c.best_graph_output_dir={tempdir}').split()

        if verbose:
            print(cmd)

        subprocess.check_call(cmd, cwd=quarl_dir)
        qc = QuantumCircuit.from_qasm_file(str(output_qasm_file))

    return qc


def quartz_optimize(qc: QuantumCircuit, gate_set, ecc_file, verbose=True) -> QuantumCircuit:
    ecc_file = Path(ecc_file)
    assert ecc_file.is_file(), ecc_file

    from quartz import Graph, Context
    from qiskit.qasm2 import dumps

    input_qasm = dumps(qc)
    context = Context(gate_set, ecc_file, verbose=verbose)
    g = Graph.from_qasm_str(context, input_qasm, verbose=verbose)
    g = g.greedy_optimize(ecc_file, verbose=verbose)
    qasm_str = g.to_qasm_str()

    return QuantumCircuit.from_qasm_str(qasm_str)


def transpile_circuit(
        circuit_path: Path,
        gate_set: List[str],
        graph_model: Union[str, nx.Graph],
        opt_params: dict = None,
        opt_method: OptMethod = OptMethod.PASSES,
        layout_method: LayoutMethod = LayoutMethod.SABRE,
        routing_method: RoutingMethod = RoutingMethod.SABRE,
        opt_order: OptOrder = OptOrder.AFTER_ROUTING,  # Always use 'after'
        verbose=True,
        verify_circuit=True,
):
    """Transpile to a physical coupling graph without optimization (right now)"""
    if opt_params is None:
        opt_params = {}
    opt_params.update(gate_set=gate_set)

    circuit_path = Path(circuit_path)
    assert circuit_path.is_file(), circuit_path

    # Load the input circuit.
    qc_input = qc = QuantumCircuit.from_qasm_file(str(circuit_path))
    num_qubits = qc_input.num_qubits
    if verify_circuit:
        if num_qubits > 10:
            print('Turn off verify_circuit because qubits is', num_qubits)
            verify_circuit = False

    if isinstance(graph_model, str):
        coupling_map, graph_name = generate_graph_for_num_qubits(graph_model, num_qubits=num_qubits)
    elif isinstance(graph_model, IBMQHardwareArchitecture):
        coupling_map = graph_model
        graph_name = graph_model.name
    else:
        raise TypeError(graph_model)

    # Before routing, do xfers.
    if opt_order == OptOrder.BEFORE_ROUTING:
        qc = optimize_circuit(qc=qc, opt_method=opt_method, opt_params=opt_params, verbose=verbose)

    if opt_method.value.startswith('qiskit:'):
        optimization_level = int(opt_method.value.split(':')[-1])
    else:
        optimization_level = 0

    # Perform routing & layout.
    qc_output = qc = route_circuit(qc=qc, coupling_map=coupling_map, routing_method=routing_method,
                              layout_method=layout_method, gate_set=gate_set, optimization_level=optimization_level)

    # After routing, do xfers.
    if opt_order == OptOrder.AFTER_ROUTING:
        qc_output = optimize_circuit(qc=qc, opt_method=opt_method, opt_params=opt_params, verbose=verbose)

    if verify_circuit:
        assert verify_circuit_equivalent(qc_input, qc_output)

    result = dict(
        circuit=circuit_path.stem,
        opt_order=opt_order.value,
        opt_method=opt_method.value,
        layout_method=layout_method.value,
        routing_method=routing_method.value,
        graph_name=graph_name,
        **qknob_metrics(qc_input, qc_output),
    )

    return result
