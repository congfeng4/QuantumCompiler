from script.maskable_ppo import run_vec_env


if __name__ == '__main__':
    run_vec_env(
        data_name='20Q_gate_Tokyo',
        mode='lstm',
        embed_dim=128,
    )
