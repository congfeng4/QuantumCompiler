"""
Search the final-reward param space.
"""

from contrib.maskable_ppo import run_maskable_ppo, get_circuits


if __name__ == '__main__':
    for path in get_circuits():
        for final_reward in [0.1, 1, 10, 100]:
            run_maskable_ppo(
                hardware='Tokyo',
                circuit_path=path,
                final_reward=final_reward,
                output_dirname='short_20Q_final_reward',
            )
