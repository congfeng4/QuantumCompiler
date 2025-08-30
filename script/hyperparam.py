from contrib.common import Unit, get_cnot_num, read_circuit
from contrib.environs import TopologicalOrderMode
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset, forward_backward_initial_mapping
from contrib.feature_extractor import QubitEmbeddingMode, PositionalEncodingMode

if __name__ == '__main__':
    mode = 'transformer'
    num_circuits = 1
    num_epochs = 100

    dataset = CircuitDataset('20Q_gate_Tokyo', sort=True, shuffle=False)

    hardware = dataset.hardware
    
    for path in dataset.sample(num_circuits=num_circuits, min_gatelen=0, 
                               max_gatelen=50):
        qc = read_circuit(path)
        init = get_initial_mapping(qc, hardware, InitialMappingStrategy.SABRE)

        run_maskable_ppo(
            hardware=hardware,
            circuit_path=qc,
            seqlen=0.5,
            output_dirname=f'len50',
            reward_shaping_weight=10,
            mode=mode,
            save_result=True,
            skip_existing=False,
            num_epochs=num_epochs,
            batch_size=4 * Unit.K,
            eval_freq=4 * Unit.K,
            n_steps=Unit.K,
            init_strategy=init,
            num_envs=16,
            nhead=2,
            num_layers=4,
            embed_dim=128,
            topological_order_mode=TopologicalOrderMode.LEVEL_ORDER,
            pe_mode=PositionalEncodingMode.LEVEL_PE,
        )
