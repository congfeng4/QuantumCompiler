import subprocess
from typing import Type, Union, List
import networkx as nx
from pathlib import Path
from qiskit import transpile, generate_preset_pass_manager
from qiskit import QuantumCircuit
from dataclasses import dataclass
import sys
from enum import Enum
import tempfile
import time
from joblib import Parallel, delayed
import pandas as pd
from pathlib import Path

from contrib.common import dict_product

from qiskit.transpiler import CouplingMap

from contrib.initial_mapping import InitialMappingStrategy
from contrib.random_graphs import generate_graph_for_num_qubits, create_coupling_graph
from contrib.common import qknob_metrics, get_total_ops, get_cnot_num, get_gate_set, write_circuit, \
    convert_to_int_mapping, write_mapping, read_mapping
from contrib.verify_circuit import check_equivalence, check_circuits_correctness
from hamap import IBMQHardwareArchitecture
from hamap.mapping import ha_mapping
from contrib.initial_mapping import get_initial_mapping

SUPPORTED_GRAPH_MODEL = ('line', 'star', 'grid', 'ring')
SUPPORTED_OPT_LEVEL = tuple(range(3))


class OptOrder(Enum):
    BEFORE_ROUTING = 'before'
    AFTER_ROUTING = 'after'
    BOTH = 'both'


class OptMethod(Enum):
    NONE = 'none'
    QUARL = 'quarl'
    QUARTZ = 'quartz'
    QISKIT_LV1 = 'qiskit:1'
    QISKIT_LV2 = 'qiskit:2'
    QISKIT_LV3 = 'qiskit:3'


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
SUPPORTED_OPT_METHOD = (OptMethod.NONE,
                        OptMethod.QISKIT_LV1, OptMethod.QISKIT_LV2, OptMethod.QISKIT_LV3)


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


def get_circuit_and_name(circuit_path: Union[QuantumCircuit, str, Path]):

    if isinstance(circuit_path, QuantumCircuit):
        qc = circuit_path
        name = qc.name
    elif isinstance(circuit_path, (Path, str)):
        circuit_path = Path(circuit_path)
        name = circuit_path.name
        assert circuit_path.is_file(), circuit_path
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
    else:
        raise TypeError(circuit_path)
    return qc, name


def route_circuit(qc: QuantumCircuit,
                  coupling_map,
                  routing_method: RoutingMethod,
                  layout_method: LayoutMethod):

    if routing_method == RoutingMethod.HA:
        initial_mapping = get_initial_mapping(qc, coupling_map, layout_method.to_initial_mapping_strategy())
        qc_output, final_mapping = ha_mapping(qc, initial_mapping=initial_mapping, hardware=coupling_map)
        return qc_output

    if routing_method in (RoutingMethod.SABRE, RoutingMethod.BASIC):
        if isinstance(coupling_map, nx.Graph):
            coupling_map = create_coupling_graph(coupling_map)
        qc = qiskit_routing(qc, coupling_map, routing_method, layout_method)
        return qc

    raise ValueError(routing_method)


def optimize_circuit(qc: QuantumCircuit, opt_method: OptMethod, opt_params: dict = None, verbose=False):
    if opt_method == OptMethod.NONE:
        return qc

    if opt_method == OptMethod.QUARL:
        return quarl_optimize(qc,
                              ecc_file=opt_params['ecc_file'],
                              quarl_dir=opt_params['quarl_dir'],
                              verbose=verbose,
                              )

    if opt_method == OptMethod.QUARTZ:
        return quartz_optimize(qc,
                               quarl_dir=opt_params['quarl_dir'],
                               ecc_file=opt_params['ecc_file'],
                               verbose=verbose)

    if opt_method.value.startswith('qiskit:'):  # Router should optimize it.
        optimization_level = int(opt_method.value.split(':')[-1])
        assert optimization_level > 0, optimization_level
        return qiskit_optimize(qc, optimization_level)

    raise ValueError(opt_method)


def quarl_optimize(qc: QuantumCircuit,
                   ecc_file: Path,
                   quarl_dir: Path,
                   max_iterations=50,
                   verbose=True) -> QuantumCircuit:
    ecc_file = Path(ecc_file)
    if not ecc_file.is_absolute():
        ecc_file = quarl_dir / ecc_file
    assert ecc_file.is_file(), ecc_file
    assert quarl_dir.exists(), quarl_dir

    from qiskit.qasm2 import dumps

    gate_set = set(get_gate_set(qc) + ['neg', 'x', 'add'])  # Assume we don't need to docompose our gates.

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

        subprocess.check_call(cmd, cwd=quarl_dir / 'experiment/ppo-new')
        qc = QuantumCircuit.from_qasm_file(str(output_qasm_file))

    return qc


def quartz_optimize(qc: QuantumCircuit, quarl_dir: Path, ecc_file, verbose=True) -> QuantumCircuit:
    ecc_file = Path(ecc_file)
    if not ecc_file.is_absolute():
        ecc_file = quarl_dir / ecc_file
    assert ecc_file.is_file(), ecc_file

    from quartz import Graph, Context
    from qiskit.qasm2 import dumps

    input_qasm = dumps(qc)
    gate_set = set(get_gate_set(qc) + ['neg', 'x', 'add'])  # Assume we don't need to docompose our gates.
    print(f'{gate_set=}')
    context = Context(list(gate_set), ecc_file, verbose=verbose)
    g = Graph.from_qasm_str(context, input_qasm)
    g = g.greedy_optimize(ecc_file, verbose=verbose)
    qasm_str = g.to_qasm_str()

    return QuantumCircuit.from_qasm_str(qasm_str)


def ha_routing(qc: QuantumCircuit, hardware: IBMQHardwareArchitecture, layout_method: LayoutMethod):
    initial_mapping = get_initial_mapping(qc, hardware, layout_method.to_initial_mapping_strategy())
    qc, _ = ha_mapping(qc, initial_mapping=initial_mapping, hardware=hardware)
    return qc


def qiskit_routing(qc: QuantumCircuit, cm: CouplingMap, routing_method: RoutingMethod, layout_method: LayoutMethod):
    assert routing_method != RoutingMethod.HA
    # Layout and Routing with qiskit.
    return transpile(
        qc,
        coupling_map=cm,
        optimization_level=0,
        layout_method=layout_method.value,
        routing_method=str(routing_method.value),
    )


def qiskit_optimize(qc: QuantumCircuit, level: int):
    pm = generate_preset_pass_manager(optimization_level=level).optimization
    return pm.run(qc)


def get_graph_and_name(graph_model: str, num_qubits: int):
    cm, graph_name = generate_graph_for_num_qubits(graph_model, num_qubits=num_qubits,
                                                                 return_coupling_map=False)
    edges = list(cm.edges)

    return cm, edges, graph_name


def transpile_circuit(
        circuit_path: Union[Path, str, QuantumCircuit],
        graph_model: Union[str, nx.Graph],
        opt_order: OptOrder,
        opt_method: OptMethod = OptMethod.QISKIT_LV2,
        layout_method: LayoutMethod = LayoutMethod.SABRE,
        routing_method: RoutingMethod = RoutingMethod.SABRE,
        opt_params: dict = None,
        verbose=True,
        verify_circuit=True,
):
    """Transpile to a physical coupling graph without optimization (right now)"""
    if opt_params is None:
        opt_params = {}

    # Load the input circuit.
    qc, qc_name = get_circuit_and_name(circuit_path)
    qc_input = qc
    num_qubits = qc_input.num_qubits

    coupling_map, edges, graph_name = get_graph_and_name(graph_model, num_qubits)

    if opt_order in (OptOrder.BOTH, OptOrder.BEFORE_ROUTING):
        qc = optimize_circuit(qc=qc, opt_method=opt_method, opt_params=opt_params, verbose=verbose)

    # Perform routing & layout.
    qc = route_circuit(qc=qc,
                       coupling_map=coupling_map, routing_method=routing_method,
                       layout_method=layout_method)

    # After routing, do xfers.
    if opt_order in (OptOrder.BOTH, OptOrder.AFTER_ROUTING):
        qc = optimize_circuit(qc=qc, opt_method=opt_method, opt_params=opt_params, verbose=verbose)

    qc_output = qc

    if verify_circuit:
        if not check_circuits_correctness(qc_input, qc_output, edges):
            return None

    result = dict(
        circuit=qc_name,
        opt_method=opt_method.value,
        layout_method=layout_method.value,
        routing_method=routing_method.value,
        opt_order=opt_order.value,
        graph_name=graph_name,
        **qknob_metrics(qc_input, qc_output),
    )

    return result


def run_baseline(
        circuit_paths: list,
        opt_method: list,
        routing_method: list,
        layout_method: list,
        graph_model: list,
        opt_order: list,
        save_file: Path = None,
        n_jobs=-1,
        opt_params=None,
        verbose=True,
):
    rounds = dict_product(dict(
        circuit_path=circuit_paths,
        opt_method=opt_method,
        routing_method=routing_method,
        graph_model=graph_model,
        layout_method=layout_method,
        opt_order=opt_order,
    ))

    results = Parallel(n_jobs=n_jobs, verbose=999)(delayed(transpile_circuit)(
        opt_params=opt_params,
        **rnd,
    ) for rnd in rounds)

    df = pd.DataFrame.from_records(filter(None, results))

    if save_file is None:
        save_file = f'./tranpile-result-{time.time()}.csv'

    save_file = Path(save_file)
    save_dir = save_file.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_file, index=False)
    if verbose:
        print(f"Done")
