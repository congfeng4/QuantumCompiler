import random
import itertools
from cgi import maxlen
from pathlib import Path

from qiskit import QuantumCircuit

from contrib.common import get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, get_short_20Q_circuits


def get_param_space(rs_weights: list[float], final_rewards: list[int], horizon_lens: list[int]):
    for rs, fr, hl in itertools.product(rs_weights, final_rewards, horizon_lens):
        yield dict(rs_weight=rs, final_reward=fr, horizon_len=hl)


if __name__ == '__main__':
    for path in get_short_20Q_circuits(min_gatelen=0, max_gatelen=9999):
        run_maskable_ppo(
            hardware='Tokyo',
            circuit_path=path,
            seqlen=64,
            final_reward=100,
            output_dirname='20Q_gate_seqlen=64_all',
            save_result=True,
            skip_existing=True,
            total_timesteps=200_000,
        )
