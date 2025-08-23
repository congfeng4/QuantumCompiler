from contrib.common import Unit
from contrib.maskable_ppo import run_maskable_ppo, get_circuits


if __name__ == '__main__':
    mode = 'transformer'
    seqlen = 16
    num_envs = None
    n_steps = 16 * Unit.K
    eval_freq = Unit.K
    batch_size = Unit.K

    num_epochs = 100
    
    for path in get_circuits(min_gatelen=200, max_gatelen=300):
        run_maskable_ppo(
            hardware='Tokyo',
            circuit_path=path,
            seqlen=seqlen,
            final_reward='cx_num',
            output_dirname=f'tmp',
            reward_shaping_weight=10,
            mode=mode,
            save_result=True,
            skip_existing=True,
            num_epochs=num_epochs,
            batch_size=batch_size,
            n_steps=n_steps,
            eval_freq=eval_freq,
        )
