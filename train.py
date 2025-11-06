from pathlib import Path

from contrib.maskable_ppo import run_maskable_ppo
from contrib.initial_mapping import InitialMappingStrategy
from argparse import ArgumentParser

from hamap import IBMQHardwareArchitecture

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', '-p', help='circuit path', type=Path)
    parser.add_argument('--feature_dim', '-f', default=64, type=int)
    parser.add_argument('--hardware', '-w', type=str)
    parser.add_argument('--layout', '-i', type=InitialMappingStrategy)
    parser.add_argument('--output', '-o', type=Path, help='output dir')
    parser.add_argument('--n_envs', '-n', type=int, default=8, help='number of envs')

    args = parser.parse_args()

    run_maskable_ppo(
        init_strategy=args.layout,
        feature_dim=args.feature_dim,
        hardware=args.hardware,
        circuit_path=args.path,
        num_envs=args.n_envs,
        n_eval_episodes=4,
        max_no_improvement_evals=4,
        output_dir=args.output,
        verbose=False,
    )
