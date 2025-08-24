from contrib.common import Unit, get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'gru'
    seqlen = 16
    n_steps = 4 * Unit.K
    eval_freq = Unit.K
    batch_size = 1024
    num_circuits = 1

    dataset = CircuitDataset('20Q_gate_Tokyo')
    min_gatelen, max_gatelen = dataset.cx_num_range
    maxlen = 200
    num_epochs = maxlen
    minlen = maxlen - 100
    
    for path in dataset.sample(5, minlen, maxlen):
        run_maskable_ppo(
            hardware='Tokyo',
            circuit_path=path,
            seqlen=seqlen,
            output_dirname=f'20Q_gate_Tokyo_maxlen=200',
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
