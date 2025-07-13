"""
Create expert trajectories from actions and save them
"""
import numpy as np
from imitation.data.types import TrajectoryWithRew, Transitions
from imitation.data import serialize

from contrib.common import RESULT_DIR, get_hardware_name
from script.benchmark_ha_qknob import HA_RESULT_DIR
from concurrent.futures import ProcessPoolExecutor
from contrib.environs import CircuitEnvWithInitialMapping, to_imitation_trajectory
from pathlib import Path
import json


def func2(traj_path: Path):
    traj_data = json.loads(traj_path.read_text())
    if 'hardware_name' not in traj_data:
        print(f'RM {traj_data}')
        traj_path.unlink(missing_ok=True)


def func(traj_path: Path):
    traj_data = json.loads(traj_path.read_text())
    traj_data = CircuitEnvWithInitialMapping.apply_trajectory(traj_data)
    print(f'Expert trajectory: {traj_path}')
    hardware_name = traj_data['hardware_name']
    return to_imitation_trajectory(traj_data), hardware_name


IMI_RESULT_DIR =  RESULT_DIR / 'expert'


if __name__ == '__main__':
    with ProcessPoolExecutor(4) as executor:
        expert_trajs_with_hardware = list(executor.map(func, HA_RESULT_DIR.glob("*.json")))

    from collections import defaultdict

    expert_trajs = defaultdict(list)
    for traj, hardware_name in expert_trajs_with_hardware:
        expert_trajs[hardware_name].append(traj)

    for hardware_name, trajs in expert_trajs.items():
        serialize.save(IMI_RESULT_DIR / f'expert-{hardware_name}.trajs', trajs)
