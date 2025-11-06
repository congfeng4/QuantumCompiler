from contrib.baselines import *


if __name__ == '__main__':
    circuit_paths = list(Path('./data/nam_circs').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=[OptMethod.NONE, OptMethod.QISKIT_LV1, OptMethod.QISKIT_LV2, OptMethod.QISKIT_LV3],
        routing_method=[RoutingMethod.SABRE],
        layout_method=[LayoutMethod.SABRE],
        opt_order=[OptOrder.BOTH, OptOrder.BEFORE_ROUTING,
                   OptOrder.AFTER_ROUTING],
        graph_model=SUPPORTED_GRAPH_MODEL,
        save_file=Path(f'./output/baseline/qiskit_name_circs.csv'),
    )
