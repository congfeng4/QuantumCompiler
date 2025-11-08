from contrib.baselines import *


if __name__ == '__main__':
    circuit_paths = list(Path('./data/nam_circs').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=[OptMethod.QISKIT_LV3],
        routing_method=[RoutingMethod.SABRE],
        layout_method=[LayoutMethod.SABRE],
        opt_order=[OptOrder.BOTH, OptOrder.BEFORE_ROUTING,
                   OptOrder.AFTER_ROUTING],
        graph_model=['ring', 'star', 'line', 'grid'],
        save_file=Path(f'./output/baseline/qiskit_name_circs.csv'),
    )
