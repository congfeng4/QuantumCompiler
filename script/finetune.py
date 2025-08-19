import itertools
from contrib.maskable_ppo import run_maskable_ppo, get_short_20Q_circuits


def get_param_space(rs_weights: list[float], final_rewards: list[int], horizon_lens: list[int]):
    for rs, fr, hl in itertools.product(rs_weights, final_rewards, horizon_lens):
        yield dict(rs_weight=rs, final_reward=fr, horizon_len=hl)


if __name__ == '__main__':
    mode = 'gru'
    seqlen = 0.1

    for path in get_short_20Q_circuits(min_gatelen=0, max_gatelen=9999):
    # path = '../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_5_1.5_no.3.qasm'
        run_maskable_ppo(
            hardware='Tokyo',
            circuit_path=path,
            seqlen=seqlen,
            final_reward='cx_num',
            output_dirname=f'20Q_gate_seqlen={seqlen}_mode={mode}',
            reward_shaping_weight=10,
            mode=mode,
            save_result=True,
            skip_existing=True,
            total_timesteps=200_000,
        )
