from contrib.common import Unit
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy

if __name__ == '__main__':
    mode = 'transformer'
    seqlen = 0.5
    batch_size = Unit.K
    num_circuits = 5
    n_steps = 16 * Unit.K
    eval_freq = 16 * Unit.K
    total_timesteps = 800 * Unit.K

    dataset = CircuitDataset('20Q_gate_Tokyo', sort=True)

    for path in dataset.circuit_paths:
        *_, metrics = run_maskable_ppo(
            init_strategy=InitialMappingStrategy.SABRE,
            feature_dim=128,
            hardware=dataset.hardware,
            circuit_path=path,
            output_dirname=dataset.dataname,
            mode=mode,
            save_result=True,
            skip_existing=True,
            total_timesteps=total_timesteps,
            batch_size=batch_size,
            eval_freq=eval_freq,
            n_steps=n_steps,
            num_envs=1,
            nhead=4,
            num_layers=8,
            max_no_improvement_evals=4,
        )

