from contrib.common import Unit, get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'gru'
    seqlen = 16
    n_steps = 128 * Unit.K
    eval_freq = Unit.K
    batch_size = Unit.K
    num_circuits = 1

    dataset = CircuitDataset('20Q_gate_Tokyo')
    min_gatelen, max_gatelen = dataset.cx_num_range
    num_epochs = 100

    for path in dataset.sample(num_circuits=num_circuits, min_gatelen=100, 
                               max_gatelen=200):
        print(path)

        run_maskable_ppo(
            hardware='Tokyo',
            circuit_path=path,
            seqlen=seqlen,
            output_dirname=f'n_step={n_steps}',
            reward_shaping_weight=10,
            mode=mode,
            save_result=True,
            skip_existing=True,
            num_epochs=num_epochs,
            batch_size=batch_size,
            n_steps=n_steps,
            eval_freq=eval_freq,
            stop_if_no_improvement=True,
        )
