from pathlib import Path
from contrib.baselines import SUPPORTED_GRAPH_MODEL
from contrib.common import Unit
from contrib.maskable_ppo import run_maskable_ppo
from contrib.initial_mapping import InitialMappingStrategy
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', '-p', help='circuit path', type=Path,
                        default=Path('./data/nam_circs/barenco_tof_3.qasm'))
    parser.add_argument('--feature_dim', '-f', default=64, type=int)
    parser.add_argument('--hardware', '-w', type=str, default='line',
                        choices=SUPPORTED_GRAPH_MODEL + ('tokyo', 'sycamore', 'rochester'))
    parser.add_argument('--layout', '-i', type=InitialMappingStrategy, default=InitialMappingStrategy.SABRE)
    parser.add_argument('--output', '-o', type=Path, help='output dir', default=Path('./output/test'))
    parser.add_argument('--n_envs', '-n', type=int, default=6, help='number of envs')
    parser.add_argument('--total_timesteps', '-t', type=int, default=100, help='number of K total timesteps')

    args = parser.parse_args()

    run_maskable_ppo(
        init_strategy=args.layout,
        feature_dim=args.feature_dim,
        hardware=args.hardware,
        circuit_path=args.path,
        num_envs=args.n_envs,
        output_dir=args.output,
        total_timesteps=args.total_timesteps * Unit.K,
        n_eval_episodes=4,
        max_no_improvement_evals=4,
        verbose=False,
    )
