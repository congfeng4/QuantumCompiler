from contrib.baselines import *


if __name__ == '__main__':
    circuit_paths = list(Path('./data/nam_circs').glob("*.qasm"))

    run_baseline(
        circuit_paths,
        opt_method=[OptMethod.QUARL],
        routing_method=SUPPORTED_ROUTING_METHOD,
        layout_method=SUPPORTED_LAYOUT_METHOD,
        graph_model=SUPPORTED_GRAPH_MODEL,
        save_file=Path(f'./output/baseline/nam_circs_qrl.csv'),
        n_jobs=1,
        opt_params=dict(
            ecc_file=Path('experiment/ecc_set/nam_325_ecc.json'),
            quarl_dir=Path('/home/mscs/congfeng4/Quantum/Quarl-artifact-master'),
        )
    )
