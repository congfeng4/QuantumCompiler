import json

from pathlib import Path
from contrib.common import IBMQHardwareArchitecture
from contrib.environs import BaselineMode, RewardMode
from contrib.maskable_ppo import create_vec_env_from_circuits, run_maskable_ppo


if __name__ == '__main__':
    bs = 128
    ns = 1000
    embed_dim = 128
    L = 8
    ent_coef = 0.01
    output_dirname = 'hcost_baseline'
    baseline_mode = BaselineMode.SUBTRACT_AVG
    reward_mode = RewardMode.HEURISTIC_COST
    gamma = 0.99

    circuit_list = list(Path('../data/53Q_gate_Sycamore/circuits/').glob('*.qasm'))
    circuit_path = circuit_list[0]
    hardware = IBMQHardwareArchitecture('Sycamore')

    env = create_vec_env_from_circuits([str(circuit_path)], hardware, num=1, add_sabre=True, L=L,
                                       baseline_mode=baseline_mode, reward_mode=reward_mode)
    circuit_name = Path(circuit_path).stem

    log_name = (f'Q={circuit_name}-B={bs}-NS={ns}-E={ent_coef}-D={embed_dim}-L={L}-'
                f'BM={baseline_mode.value}-RM={reward_mode.value}-GA={gamma}')

    metrics = env.get_attr('metrics_baseline', 0)
    print(circuit_name, 'HA', metrics)

    run_maskable_ppo(
        log_name=log_name,
        hardware=hardware,
        env=env,
        embed_dim=embed_dim,
        batch_size=bs,
        gamma=gamma,
        n_steps=ns,
        seqlen=L,
        mode='gru',
        ent_coef=ent_coef,
        total_timesteps=200_000,
        output_dirname=output_dirname,
        early_stop=False,
    )
    metrics = env.get_attr('metrics', 0)
    print(circuit_name, 'HA & PPO', metrics)

