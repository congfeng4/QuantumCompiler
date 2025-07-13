"""
Create expert trajectories from actions and save them
"""
import pickle

from imitation.data import serialize, rollout

from contrib.common import RESULT_DIR
from concurrent.futures import ProcessPoolExecutor
from contrib.ha_traj import load_and_group_trajectories
from contrib.environs import CircuitEnvWithInitialMapping, to_imitation_trajectory
from pathlib import Path
import json


def func(hardware_name: str, traj_paths: list[Path]):
    expert_trajs = []

    for traj_path in traj_paths:
        traj_data = json.loads(traj_path.read_text())
        traj_env = CircuitEnvWithInitialMapping.apply_trajectory(traj_data)
        traj = to_imitation_trajectory(traj_env)
        expert_trajs.append(traj)

    # Save trajectories.
    save_file = EXPERT_WITH_INIT_DIR / f'{hardware_name}.trajs'
    serialize.save(save_file, expert_trajs)

    # Save transitions (flatten very slow)
    expert_trajs = rollout.flatten_trajectories(expert_trajs)
    save_file = EXPERT_WITH_INIT_DIR / f'{hardware_name}.trans'
    with save_file.open('wb') as f:
        pickle.dump(expert_trajs, f)

    print(f'Expert trajectory: {hardware_name}')


EXPERT_WITH_INIT_DIR = RESULT_DIR / 'expert' / 'with_init'

def _func(args): return func(*args)


if __name__ == '__main__':
    with ProcessPoolExecutor(1) as executor:
        list(executor.map(_func, load_and_group_trajectories().items()))
