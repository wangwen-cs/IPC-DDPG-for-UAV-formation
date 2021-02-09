import time

import numpy as np
import torch
from train_ddpg import DDPGActor
from uav_2d import ManyUavEnv


class DDPGArgs:

    def __init__(self):
        self.exp_cap = 1000000
        self.gamma = 0.95
        self.batch = 128
        self.test_interval = 32
        self.update_cnt = 200
        self.update_interval = 200
        self.actor_lr = 1e-4
        self.critic_lr = 5e-3

        self.env_name = 'UAV_1'
        self.action_bound = [np.pi / 4, 1.0]
        self.max_ep = 100000
        self.scale = 1.0

        self.log_dir = './logs/ddpg/{}'.format(self.env_name)


def choose_action(actor_net, state):
    with torch.no_grad():
        state = torch.from_numpy(state).float()
        action = actor_net(state)
    action = action.detach().numpy()
    return action


def eval_model(model_name):
    actor = DDPGActor(16, 2)
    actor.load_state_dict(torch.load(model_name))

    env = ManyUavEnv(1, False)
    done = False
    state = env.reset()
    while not done:
        action = choose_action(actor, state)
        state_, reward, done, _ = env.step(action * [np.pi / 4, 1.0])
        print(reward)
        env.render()
        state = state_
    env.close()


def _main():
    eval_model('./result/4001.pkl')


if __name__ == '__main__':
    _main()


