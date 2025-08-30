from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGNode, DAGOpNode, DAGInNode

from contrib.common import Unit, get_cnot_num, read_circuit
from contrib.environs import CircuitEnvWithInitialMapping, build_op_node_level
from contrib.maskable_ppo import run_maskable_ppo, CircuitDataset


if __name__ == '__main__':
    mode = 'transformer'
    seqlen = 16
    batch_size = Unit.K
    num_circuits = 5
    n_steps = 16 * Unit.K
    eval_freq = 16 * Unit.K

    dataset = CircuitDataset('20Q_gate_Tokyo')

    path = next(dataset.sample(1, 0, 50))
    qc = read_circuit(path)
    print('cx num', get_cnot_num(qc))
    dag = circuit_to_dag(qc)
    topo = list(filter(lambda nd: nd.name == 'cx', dag.topological_op_nodes()))
    print('topo', [op._node_id for op in topo])

    import rustworkx as rx
    topo2 = list(filter(lambda nd: isinstance(nd, DAGOpNode) and nd.name == 'cx', map(lambda idx: dag._multi_graph.nodes()[idx],
                                                        rx.topological_sort(dag._multi_graph))))
    print('topo2', [op._node_id for op in topo2])

    memo = build_op_node_level(dag, topo)
    #
    # print('topo', [op._node_id for op in topo])
    print('memo', memo)
    # print(f'{memo.keys()=}')
    # print(f'{memo.values()=}')

    # for maxlen in range(100, 700, 100):
        # minlen = maxlen - 100
        # num_epochs = 50 * (maxlen // 100)
        #
        # for path in dataset.sample(num_circuits, minlen, maxlen):
        #     run_maskable_ppo(
        #         embed_dim=128,
        #         hardware='Tokyo',
        #         circuit_path=path,
        #         seqlen=seqlen,
        #         output_dirname=f'20Q_gate_Tokyo_level_PE',
        #         reward_shaping_weight=10,
        #         mode=mode,
        #         save_result=True,
        #         skip_existing=False,
        #         num_epochs=num_epochs,
        #         batch_size=batch_size,
        #         eval_freq=eval_freq,
        #         n_steps=n_steps,
        #         num_envs=16,
        #         nhead=4,
        #         num_layers=8,
        #     )
