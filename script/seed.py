import os
import random
import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv

GLOBAL_SEED = 42          # 统一只改这一行即可

def set_all_seeds(seed: int = GLOBAL_SEED, env=None):
    """把全局随机源全部锁死，可选把已建 VecEnv 也一次性 seed。"""
    # 1. 操作系统级哈希种子
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. Python 内置 random
    random.seed(seed)

    # 3. NumPy
    np.random.seed(seed)

    # 4. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 所有 GPU

    # 5. 强制 CUDA 确定性（牺牲少量速度换可复现）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 6. Stable-Baselines3 统一接口
    set_random_seed(seed, using_cuda=torch.cuda.is_available())

    # 7. 如果已经创建了 VecEnv，再补一次
    if env is not None and isinstance(env, VecEnv):
        env.seed(seed)


if __name__ == '__main__':
    set_all_seeds()