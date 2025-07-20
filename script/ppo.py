"""
直接用PPO是很难收敛的，因为非法动作空间十分巨大。
至少需要用MaskablePPO，并且把Action Mask定义好。
"""
from stable_baselines3.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from contrib.action import ActionAsPolicyInt, ActionAsPolicyCandidates
from contrib.environs import *
from contrib.ha_traj import get_initial_mapping, InitialMappingStrategy


if __name__ == '__main__':
    bs = 128
    ns = 5000
    log_name = f"tokyo_ppo_cand_bs={bs}_ns={ns}"

    rng = np.random.default_rng(0)
    env = CircuitEnvWithInitialMapping.make(
        input_circuit_path='../data/20Q_gate_Tokyo/circuits/20Q_gate_Tokyo_large_1_10_1.5_no.1.qasm',
        hardware_name='tokyo',
        init=InitialMappingStrategy.IDENTITY,
        a2p=ActionAsPolicyCandidates(),
    )

    ppo = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=ns,
        batch_size=bs,
        tensorboard_log="../log/",
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(
                vf=[512, 256, 128],
                pi=[512, 256, 128],
            )
        )
    ).learn(
        total_timesteps=1_000_000,
        tb_log_name=log_name,
    )

    print('Eval policy')
    reward, _ = evaluate_policy(ppo, env, 10)
    print("Reward:", reward)
