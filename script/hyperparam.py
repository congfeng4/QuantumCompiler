from contrib.common import Unit, get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'transformer'
    seqlen = 32
    n_steps = 16 * Unit.K
    eval_freq = 16 * Unit.K
    batch_size = Unit.K
    num_circuits = 1

    dataset = CircuitDataset('20Q_gate_Tokyo', sort=True, shuffle=False)
    num_epochs = 50

    for path in dataset.sample(num_circuits=num_circuits, min_gatelen=100, 
                               max_gatelen=150):
        
        for num_layers in [8, 4, 2]:
            
            for nhead in [8, 4, 2]:
                
                for seqlen in [64, 32, 16]:

                    run_maskable_ppo(
                        hardware='Tokyo',
                        circuit_path=path,
                        seqlen=seqlen,
                        output_dirname=f'num_layers_nhead_len100',
                        reward_shaping_weight=10,
                        mode=mode,
                        save_result=True,
                        skip_existing=False,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        eval_freq=eval_freq,
                        n_steps=n_steps,
                        num_envs=16,
                        nhead=nhead,
                        num_layers=num_layers,
                    )
