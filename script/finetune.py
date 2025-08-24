from contrib.common import Unit, get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'gru'
    seqlen = 16
    eval_freq = Unit.K
    batch_size = 256
    num_circuits = 3

    dataset = CircuitDataset('20Q_gate_Tokyo')
    
    for maxlen in range(200, 600, 100):
        num_epochs = maxlen
        minlen = maxlen - 100
        n_steps = 2 * maxlen // 100 * Unit.K
        
        for path in dataset.sample(num_circuits, minlen, maxlen):
            run_maskable_ppo(
                hardware='Tokyo',
                circuit_path=path,
                seqlen=seqlen,
                output_dirname=f'20Q_gate_Tokyo_sample=10',
                reward_shaping_weight=10,
                mode=mode,
                save_result=True,
                skip_existing=True,
                num_epochs=num_epochs,
                batch_size=batch_size,
                n_steps=n_steps,
                eval_freq=eval_freq,
                num_envs=24,
                learning_rate=3e-4,
            )
