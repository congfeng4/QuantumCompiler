from imitation.data import serialize, rollout
import pickle
from script.expert_trajectory import EXPERT_WITH_INIT_DIR


if __name__ == '__main__':
    trajs_path = EXPERT_WITH_INIT_DIR / 'tokyo.trajs'
    trajs = serialize.load(trajs_path)
    trans = rollout.flatten_trajectories(trajs)
    with open(EXPERT_WITH_INIT_DIR / 'tokyo.trans', 'wb') as f:
        pickle.dump(trans, f)
