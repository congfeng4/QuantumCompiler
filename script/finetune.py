import json
import random

from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

from contrib.common import IBMQHardwareArchitecture, get_cnot_num
from contrib.maskable_ppo import create_vec_env_from_circuits, run_maskable_ppo


if __name__ == '__main__':
    bs = 128
    ns = 1000
    embed_dim = 128
    ent_coef = 0.01
    max_gatelen = 100
    gamma = 0.99
    total_timesteps = 200_000
    circuit_list = list(Path('../data/20Q_gate_Tokyo/circuits/').glob('*.qasm')) + list(Path('../data/20Q_depth_Tokyo/circuits/').glob('*.qasm'))
    random.shuffle(circuit_list)
    hardware = IBMQHardwareArchitecture('Tokyo')
    mode = 'gru'
    reward_shaping_weight = 1
    L = 6
    output_dirname = f'Tokyo_all_L={L}_rs={reward_shaping_weight}'

    for circuit_path in circuit_list:
        qc = QuantumCircuit.from_qasm_file(str(circuit_path))
        gate_len = get_cnot_num(qc)
        # if gate_len <= max_gatelen:
        #     print(f'Skip {circuit_path} len {gate_len}')
        #     continue
        print(f'{circuit_path} len {gate_len} L {L}')
        env = create_vec_env_from_circuits([qc], hardware, num=1, add_sabre=True, L=L,
                                           reward_shaping_weight=reward_shaping_weight)
        circuit_name = Path(circuit_path).stem

        log_name = f'Q={circuit_name}-B={bs}-NS={ns}-E={ent_coef}-D={embed_dim}-L={L}-RS={reward_shaping_weight}'

        metrics = env.get_attr('metrics_baseline', 0)
        print(circuit_name, 'HA', metrics)

        metrics = run_maskable_ppo(
            log_name=log_name,
            hardware=hardware,
            env=env,
            embed_dim=embed_dim,
            batch_size=bs,
            gamma=gamma,
            n_steps=ns,
            mode=mode,
            ent_coef=ent_coef,
            total_timesteps=total_timesteps,
            output_dirname=output_dirname,
            early_stop=False,
        )
        print(circuit_name, 'PPO', metrics)
