import random
import itertools
from pathlib import Path

from qiskit import QuantumCircuit

from contrib.common import get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, get_short_20Q_circuits


def get_param_space(rs_weights: list[float], final_rewards: list[int], horizon_lens: list[int]):
    for rs, fr, hl in itertools.product(rs_weights, final_rewards, horizon_lens):
        yield dict(rs_weight=rs, final_reward=fr, horizon_len=hl)


if __name__ == '__main__':
    for path in get_short_20Q_circuits():
        run_maskable_ppo(
            hardware='Tokyo',
            circuit_path=path,
            seqlen=8,
            output_dirname='short_20Q_seqlen',
            save_result=False,
        )
