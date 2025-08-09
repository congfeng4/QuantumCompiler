from contrib.maskable_ppo import run_vec_env

if __name__ == '__main__':
    run_vec_env(
        data_name='20Q_gate_Tokyo',
        mode='gru',
        num_train=2,
        embed_dim=128,
        output_dirname='20Q_gate_Tokyo-gru-128-train=1',
        total_timesteps=100,
    )
