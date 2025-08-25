from contrib.common import Unit, get_cnot_num
from contrib.environs import CircuitEnvWithInitialMapping, InitialMappingCircuitEnv
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'gru'
    seqlen = 16
    eval_freq = Unit.K
    batch_size = Unit.K
    num_circuits = 10
    n_steps = 4 * Unit.K
    num_epochs = 50

    dataset = CircuitDataset('20Q_gate_Tokyo')

    for path in dataset.sample(1, 0, 50):
        run_maskable_ppo(
            env_cls=InitialMappingCircuitEnv,
            use_masking=True,
            hardware='Tokyo',
            circuit_path=path,
            seqlen=seqlen,
            output_dirname=f'mapping_debug',
            reward_shaping_weight=10,
            mode=mode,
            save_result=True,
            skip_existing=False,
            num_epochs=num_epochs,
            batch_size=batch_size,
            n_steps=n_steps,
            eval_freq=eval_freq,
            num_envs=1,
            learning_rate=3e-4,
        )
