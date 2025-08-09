import tempfile
from pathlib import Path
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy

from contrib.environs import RewardMode
from hamap import IBMQHardwareArchitecture
from contrib.maskable_ppo import create_vec_env_from_circuits

rng = np.random.default_rng(0)

hardware = IBMQHardwareArchitecture('Tokyo')

circuit_path = Path('../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_3_1.5_no.2.qasm')
env = create_vec_env_from_circuits([str(circuit_path)], hardware, num=1,
                                   add_sabre=True,
                                   add_simulated_anealing=False,
                                   add_random=False, L=16, reward_mode=RewardMode.HEURISTIC_COST)


class ExpertPolicy:

    def __init__(self, ):
        pass

    def predict(self, ):
        pass


expert = load_policy(
    "ppo-huggingface",
    organization="HumanCompatibleAI",
    env_name="seals-CartPole-v0",
    venv=env,
)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
)
with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    dagger_trainer.train(8_000)

reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
print("Reward:", reward)
