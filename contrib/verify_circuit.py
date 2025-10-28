from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap
from qiskit import transpile

from contrib.common import read_mapping, read_json
from contrib.initial_mapping import InitialMappingStrategy
from contrib.random_graphs import closest_factors

from hamap import IBMQHardwareArchitecture, apply_layout


def check_equivalence(qc_in, qc_out, initial_mapping=None):
    """
    Check functional equivalence of input and output circuits.
    """
    if initial_mapping is not None:
        qc_mapped = apply_layout(qc_in, initial_mapping)
        return Statevector(qc_mapped).equiv(Statevector(qc_out))

    # 对于qiskit transpile出来的电路，不需要Apply Layout。
    return Statevector(qc_in).equiv(Statevector(qc_out))


def check_routed(qc: QuantumCircuit, edges: list, initial_mapping=None):
    """
    Check all two-qubit gates in qc satisfy coupling map.
    """
    assert all(map(lambda e: isinstance(e, tuple) and len(e) == 2, edges))
    if initial_mapping is None:
        mapping = {qc.qubits[p]: p for p in range(qc.num_qubits)}
    else:
        # 对应qiskit出来的电路，已经ApplyLayout了，不需要自己再次映射。
        mapping = initial_mapping

    edges = set(edges)

    for inst in qc.data:  # type: CircuitInstruction
        qubits = inst.qubits
        if len(qubits) == 1:
            continue
        if len(qubits) == 2:
            u, v = qubits
            pu, pv = mapping[u], mapping[v]
            if (pu, pv) not in edges:
                print('Error:', inst.operation.name, (u, v), (pu, pv), 'not in edges')
                return False
        else:
            print('Error: only 1 and 2 qubit ops are allowed', inst)
            return False

    return True


def check_output_ours(output_dir: Path, qubits_threshold=10):
    print('Checking ours on output_dir', output_dir)

    for subdir in Path(output_dir).iterdir():
        qc_in = QuantumCircuit.from_qasm_file(subdir / 'input_circuit.qasm')
        qc_out = QuantumCircuit.from_qasm_file(subdir / 'output_circuit.qasm')
        try:
            initial_mapping = read_mapping(subdir / 'initial_mapping.json')
        except:
            initial_mapping = None  # Applied layout

        if qc_in.num_qubits > qubits_threshold:
            continue
        ok = check_equivalence(qc_in, qc_out, initial_mapping)
        if not ok:
            print('Equiv failure', subdir.name)

        try:
            edges = read_json(subdir / 'edges.json')
        except:
            print('edges missing for', subdir.name)
            continue

        ok = check_routed(qc_out, edges, initial_mapping)
        if not ok:
            print('Routing failure', subdir.name)


def check_qiskit_transpile(data_dir: Path, qubits_threshold=10):
    print('Checking qiskit transpile in data', data_dir)

    for qc_path in Path(data_dir).glob('*.qasm'):
        qc_in = QuantumCircuit.from_qasm_file(qc_path)
        num_qubits = qc_in.num_qubits
        if num_qubits > qubits_threshold:
            continue

        rows, cols = closest_factors(num_qubits)

        for name, cm in [('line', CouplingMap.from_line(num_qubits)), ('grid', CouplingMap.from_grid(rows, cols)),
                   ('ring', CouplingMap.from_ring(num_qubits)) ]:
            for routing_method in ['sabre', 'basic']:
                for layout_method in ['trivial', 'dense', 'sabre']:
                    for opt_level in range(4):
                        qc_out = transpile(qc_in, coupling_map=cm,
                                           layout_method=layout_method, routing_method=routing_method,
                                           optimization_level=opt_level)
                        assert qc_out.num_qubits == qc_in.num_qubits

                        ok = check_equivalence(qc_in, qc_out)
                        if not ok:
                            print('Equiv failure', qc_path.name, routing_method, layout_method, opt_level, name)

                        edges = [cm.graph.get_edge_endpoints_by_index(eid) for eid in cm.graph.edge_indices()]
                        ok = check_routed(qc_out, edges)
                        if not ok:
                            print('Routing failure', qc_path.name, layout_method, name)


def check_ha_mapping(data_dir: Path, qubits_threshold=10):
    print('Cheking ha-mapping in data', data_dir)

    from hamap.mapping import ha_mapping
    from contrib.initial_mapping import get_initial_mapping

    for qc_path in Path(data_dir).glob('*.qasm'):
        qc_in = QuantumCircuit.from_qasm_file(qc_path)
        num_qubits = qc_in.num_qubits
        if num_qubits > qubits_threshold:
            continue

        rows, cols = closest_factors(num_qubits)

        for name, hardware in [
            ('grid', IBMQHardwareArchitecture('grid', rows=rows, cols=cols)),
            ('line', IBMQHardwareArchitecture('line', num_nodes=num_qubits)),
            ('ring', IBMQHardwareArchitecture('ring', num_nodes=num_qubits)),
            ('star', IBMQHardwareArchitecture('star', num_nodes=num_qubits)),
        ]:
            for layout_method in [InitialMappingStrategy.SABRE, InitialMappingStrategy.IDENTITY,
                                  InitialMappingStrategy.RANDOM]:
                initial_mapping = get_initial_mapping(qc_in, hardware, layout_method)

                qc_out, _ = ha_mapping(qc_in, initial_mapping=initial_mapping, hardware=hardware)

                ok = check_equivalence(qc_in, qc_out)
                if not ok:
                    print('Equiv failure', qc_path.name, layout_method, name)

                ok = check_routed(qc_out, list(hardware.edges))
                if not ok:
                    print('Routing failure', qc_path.name, layout_method, name)


if __name__ == '__main__':
    check_ha_mapping(Path('../data/nam_circs'))
    check_qiskit_transpile(Path('../data/nam_circs'))
    check_output_ours('../output/ours')
