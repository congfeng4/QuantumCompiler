from contrib.common import Unit, get_cnot_num, read_circuit
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset, forward_backward_initial_mapping
from contrib.feature_extractor import QubitEmbeddingMode

if __name__ == '__main__':
    mode = 'transformer'
    num_circuits = 1
    num_epochs = 100

    dataset = CircuitDataset('20Q_gate_Tokyo', sort=True, shuffle=False)

    hardware = dataset.hardware
    
    for path in dataset.sample(num_circuits=num_circuits, min_gatelen=0, 
                               max_gatelen=100):
        qc = read_circuit(path)
        metrics, *_ = forward_backward_initial_mapping(qc, hardware, num_epochs=50)
        print(metrics)
        # init = get_initial_mapping(qc, hardware, InitialMappingStrategy.RANDOM)
        #
        # run_maskable_ppo(
        #     hardware=hardware,
        #     circuit_path=qc,
        #     seqlen=16,
        #     output_dirname=f'hparam_qubit_embed_mode',
        #     reward_shaping_weight=10,
        #     mode=mode,
        #     save_result=True,
        #     skip_existing=False,
        #     num_epochs=num_epochs,
        #     batch_size=Unit.K,
        #     eval_freq=16 * Unit.K,
        #     n_steps=16 * Unit.K,
        #     init_strategy=init,
        #     num_envs=16,
        #     nhead=4,
        #     num_layers=8,
        #     embed_dim=128,
        # )
