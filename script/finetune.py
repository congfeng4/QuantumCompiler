from contrib.common import Unit, get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'gru'
    seqlen = 16
    n_steps = 16 * Unit.K
    eval_freq = Unit.K
    batch_size = 1024
    num_circuits = 1

    dataset = CircuitDataset('20Q_gate_Tokyo')
    min_gatelen, max_gatelen = dataset.cx_num_range
    step_size = 100
    # num_epochs = 400

    for gatelen in range(0, 600, 100):
        print(gatelen, gatelen+step_size)
        min_gatelen=gatelen
        max_gatelen=gatelen + step_size
        num_epochs = max_gatelen
        for path in dataset.sample(num_circuits=num_circuits, min_gatelen=min_gatelen, 
                                   max_gatelen=max_gatelen):
            print(path)

            run_maskable_ppo(
                hardware='Tokyo',
                circuit_path=path,
                seqlen=seqlen,
                output_dirname=f'20Q_gate_Tokyo_num_circuits={num_circuits}_sample',
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
