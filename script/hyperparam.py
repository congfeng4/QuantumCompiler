from contrib.common import Unit, get_cnot_num
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset
from contrib.feature_extractor import QubitEmbeddingMode

if __name__ == '__main__':
    mode = 'transformer'
    num_circuits = 1
    num_epochs = 100

    dataset = CircuitDataset('20Q_gate_Tokyo', sort=True, shuffle=False)

    for path in dataset.sample(num_circuits=num_circuits, min_gatelen=100, 
                               max_gatelen=150):
        # for ent_coef in [0.1, 0.01, 0.001]:
        for qubit_embed_mode in [QubitEmbeddingMode.DISTANCE_MATRIX_MLP,
                                 QubitEmbeddingMode.GNN_EDGE_INDEX,
                                 QubitEmbeddingMode.DISTANCE_MATRIX_POS_EMBED]:
            run_maskable_ppo(
                hardware='Tokyo',
                circuit_path=path,
                seqlen=16,
                output_dirname=f'hparam_qubit_embed_mode',
                reward_shaping_weight=10,
                mode=mode,
                save_result=True,
                skip_existing=False,
                num_epochs=num_epochs,
                batch_size=Unit.K,
                eval_freq=16 * Unit.K,
                n_steps=16 * Unit.K,
                num_envs=16,
                nhead=4,
                num_layers=8,
                embed_dim=128,
                # ent_coef=0.01,
                qubit_embed_mode=qubit_embed_mode,
            )
