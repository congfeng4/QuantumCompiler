from contrib.common import Unit
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy

if __name__ == '__main__':

    dataset = CircuitDataset('20Q_gate_Tokyo', sort=False)

    for path in dataset.circuit_paths:
        run_maskable_ppo(
            init_strategy=InitialMappingStrategy.SABRE,
            feature_dim=64,
            hardware=dataset.hardware,
            circuit_path=path,
            num_envs=4,
            n_eval_episodes=4,
            max_no_improvement_evals=4,
            verbose=False,
        )

