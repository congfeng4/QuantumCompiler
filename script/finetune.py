from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode, DAGInNode

from contrib.common import Unit, get_cnot_num, read_circuit
from contrib.environs import CircuitEnvWithInitialMapping, build_op_node_level
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset
from contrib.feature_extractor import QubitEmbeddingMode, PositionalEncodingMode
from contrib.environs import TopologicalOrderMode
from contrib.initial_mapping import get_initial_mapping, InitialMappingStrategy


if __name__ == '__main__':
    mode = 'transformer'
    seqlen = 0.5
    batch_size = Unit.K
    num_circuits = 5
    n_steps = 16 * Unit.K
    eval_freq = 16 * Unit.K

    dataset = CircuitDataset('20Q_gate_Tokyo')

    for maxlen in range(50, 700, 50):
        minlen = maxlen - 50
        num_epochs = 40 * (maxlen // 50)
        
        for path in dataset.sample(num_circuits, minlen, maxlen):
            run_maskable_ppo(
                init_strategy=InitialMappingStrategy.RANDOM,
                embed_dim=128,
                hardware='Tokyo',
                circuit_path=path,
                seqlen=seqlen,
                output_dirname=f'20Q_gate_Tokyo_Level',
                reward_shaping_weight=10,
                mode=mode,
                save_result=True,
                skip_existing=False,
                num_epochs=num_epochs,
                batch_size=batch_size,
                eval_freq=eval_freq,
                n_steps=n_steps,
                num_envs=16,
                nhead=4,
                num_layers=8,
                topological_order_mode=TopologicalOrderMode.LEVEL_ORDER,
                pe_mode=PositionalEncodingMode.LEVEL_PE,
                learning_rate=1e-4,
                ent_coef=0,
            )
