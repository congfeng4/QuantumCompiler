"""
Search the seqlen param space.
"""

from contrib.maskable_ppo import run_maskable_ppo, get_circuits


if __name__ == '__main__':
    for path in get_circuits():
        for seqlen in [0.1, 0.2, 8, 16]:
            run_maskable_ppo(
                hardware='Tokyo',
                circuit_path=path,
                seqlen=seqlen,
                output_dirname='short_20Q_seqlen',
                save_result=False,
            )
