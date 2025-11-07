from contrib.baselines import *


if __name__ == '__main__':
    circuit_paths = list(Path('./data/nam_circs').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=[OptMethod.QISKIT_LV3],
        routing_method=[RoutingMethod.SABRE],
        layout_method=[LayoutMethod.SABRE],
        opt_order=[OptOrder.BOTH, OptOrder.BEFORE_ROUTING, OptOrder.AFTER_ROUTING],
        graph_model=SUPPORTED_GRAPH_MODEL,
        save_file=Path(f'./output/baseline/nam_circs_before_after.csv'),
    )

    circuit_paths = list(Path('./data/20Q_gate_Tokyo/circuits').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=[OptMethod.QISKIT_LV3],
        routing_method=[RoutingMethod.SABRE],
        layout_method=[LayoutMethod.SABRE],
        opt_order=[OptOrder.BOTH, OptOrder.BEFORE_ROUTING, OptOrder.AFTER_ROUTING],
        graph_model=['tokyo'],
        save_file=Path(f'./output/baseline/tokyo_before_after.csv'),
        basic_gates=['cx', 'u', 'h'],
    )

    circuit_paths = list(Path('./data/53Q_gate_Rochester/circuits').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=[OptMethod.QISKIT_LV3],
        routing_method=[RoutingMethod.SABRE],
        layout_method=[LayoutMethod.SABRE],
        opt_order=[OptOrder.BOTH, OptOrder.BEFORE_ROUTING, OptOrder.AFTER_ROUTING],
        graph_model=['rochester'],
        save_file=Path(f'./output/baseline/rochester_before_after.csv'),
        basic_gates=['cx', 'u', 'h'],
    )

    circuit_paths = list(Path('./data/53Q_depth_Sycamore/circuits').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=[OptMethod.QISKIT_LV3],
        routing_method=[RoutingMethod.SABRE],
        layout_method=[LayoutMethod.SABRE],
        opt_order=[OptOrder.BOTH, OptOrder.BEFORE_ROUTING, OptOrder.AFTER_ROUTING],
        graph_model=['sycamore'],
        save_file=Path(f'./output/baseline/sycamore_before_after.csv'),
        basic_gates=['cx', 'u', 'h'],
    )
