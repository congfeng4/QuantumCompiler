from contrib.common import Unit, get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'gru'
    seqlen = 16
    eval_freq = 16 * Unit.K
    batch_size = Unit.K
    num_circuits = 2
    n_steps = 16 * Unit.K
    num_epochs = 50
    clip_range = 0.2
    learning_rate=1e-4
    
    dataset = CircuitDataset('20Q_gate_Tokyo')

    for maxlen in range(100, 600, 100):
        minlen = maxlen - 100
        for path in dataset.sample(num_circuits, minlen, maxlen):
            run_maskable_ppo(
                hardware='Tokyo',
                circuit_path=path,
                seqlen=seqlen,
                output_dirname=f'20Q_gate_Tokyo_sample_100',
                reward_shaping_weight=10,
                mode=mode,
                save_result=True,
                skip_existing=False,
                num_epochs=num_epochs,
                batch_size=batch_size,
                n_steps=n_steps,
                eval_freq=eval_freq,
                num_envs=16,
                learning_rate=learning_rate,
                clip_range=clip_range,
            )
