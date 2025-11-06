from contrib.baselines import *


def run_baseline(
        circuit_paths: list,
        opt_method: list,
        routing_method: list,
        layout_method: list,
        graph_model: list,
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
    ))

    if n_jobs == 1:
        for rnd in rounds:
            transpile_circuit(opt_params=opt_params, **rnd)
        return

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
