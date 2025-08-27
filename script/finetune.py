from contrib.common import Unit, get_cnot_num
from contrib.environs import CircuitEnvWithInitialMapping
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'transformer'
    seqlen = 16
    batch_size = Unit.K
    num_circuits = 5
    n_steps = 16 * Unit.K
    eval_freq = 16 * Unit.K
    
    dataset = CircuitDataset('20Q_gate_Tokyo')

    for maxlen in range(100, 700, 100):
        minlen = maxlen - 100
        num_epochs = 50 * (maxlen // 100)
        
        for path in dataset.sample(num_circuits, minlen, maxlen):

            run_maskable_ppo(
                hardware='Tokyo',
                circuit_path=path,
                seqlen=seqlen,
                output_dirname=f'20Q_gate_Tokyo_NH=2_NL=4',
                reward_shaping_weight=10,
                mode=mode,
                save_result=True,
                skip_existing=False,
                num_epochs=num_epochs,
                batch_size=batch_size,
                eval_freq=eval_freq,
                n_steps=n_steps,
                num_envs=16,
                nhead=2,
                num_layers=4,
            )
