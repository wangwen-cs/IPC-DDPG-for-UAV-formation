import gym
import numpy as np
import torch
import ray
from torch.distributions import Normal

from ppo_args import PPOArgs
from ppo_model import PPOModel

from uav_2d_ma_fast import ManyUavEnv


@ray.remote(num_cpus=1)
class PPOWorker:

    def __init__(self, args: PPOArgs):
        self._env = ManyUavEnv(args.n_agents, np.random.randint(0, 100000), 0)
        self._observation_dim = 22
        self._action_dim = 2
        self._model = PPOModel(self._observation_dim, self._action_dim)
        self._prev_state = self._env.reset()
        self._args = args
        self._h = 0
        self._exploration_std = None

    def _choose_action(self, state):
        s_tensor = torch.tensor(state).float()
        with torch.no_grad():
            v, mu = self._model(s_tensor)
            dist = Normal(mu, self._exploration_std)
            action_tensor = dist.sample()
            action_tensor = torch.clamp(action_tensor, -1, 1, out=action_tensor)
        return action_tensor.numpy()

    def _calc_tar_adv(self, states, rewards, last_state, time_stamp, done):
        s_tensor = torch.tensor(states).float()
        next_s_tensor = torch.tensor(states[1:] + [last_state]).float()
        # a_tensor = torch.tensor(actions)
        r_tensor = torch.tensor(rewards)
        conf = self._args.gamma ** np.array(time_stamp)
        with torch.no_grad():
            v_prev, *_ = self._model(s_tensor)
            v_next, *_ = self._model(next_s_tensor)
            # old_prob = pi.gather(-1, a_tensor.view(-1, 1).long()).view(-1)
            target = r_tensor + self._args.gamma * v_next.view(-1)
            adv = target - v_prev.view(-1)
            if done:
                target[-1] = r_tensor[-1]
                adv[-1] = target[-1] - v_prev[-1]
            adv = adv * conf
        return target.numpy(), adv.numpy()

    def collect(self, n, parm, std):
        args = self._args
        states = [[] for _ in range(args.n_agents)]
        actions = [[] for _ in range(args.n_agents)]
        rewards = [[] for _ in range(args.n_agents)]
        time_stamp = [[] for _ in range(args.n_agents)]

        ret_targets = [[] for _ in range(args.n_agents)]
        ret_advantages = [[] for _ in range(args.n_agents)]
        ret_states = [[] for _ in range(args.n_agents)]
        ret_actions = [[] for _ in range(args.n_agents)]

        self._model.load_state_dict(parm)
        self._exploration_std = std

        for _ in range(n):
            action = []
            for uav_index in range(args.n_agents):
                a = self._choose_action(self._prev_state[uav_index])
                action.append(a)
            next_state, reward, done, _ = self._env.step(action)

            for uav_index in range(args.n_agents):
                states[uav_index].append(self._prev_state[uav_index])
                actions[uav_index].append(action[uav_index])
                rewards[uav_index].append(reward[uav_index])
                time_stamp[uav_index].append(self._h)

            self._prev_state = next_state
            self._h += 1
            if done:
                for uav_index in range(args.n_agents):
                    tar, adv = self._calc_tar_adv(states[uav_index], rewards[uav_index], self._prev_state[uav_index], time_stamp[uav_index], True)
                    for i, v in enumerate(states[uav_index]):
                        ret_states[uav_index].append(v)
                        ret_actions[uav_index].append(actions[uav_index][i])
                        ret_targets[uav_index].append(tar[i])
                        ret_advantages[uav_index].append(adv[i])

                
                states = [[] for _ in range(args.n_agents)]
                actions = [[] for _ in range(args.n_agents)]
                rewards = [[] for _ in range(args.n_agents)]
                time_stamp = [[] for _ in range(args.n_agents)]
                
                self._prev_state = self._env.reset()
                self._h = 0

        if len(states[0]) != 0:
            for uav_index in range(args.n_agents):
                tar, adv = self._calc_tar_adv(states[uav_index], rewards[uav_index], self._prev_state[uav_index], time_stamp[uav_index], False)
                for i, v in enumerate(states[uav_index]):
                    ret_states[uav_index].append(v)
                    ret_actions[uav_index].append(actions[uav_index][i])
                    ret_targets[uav_index].append(tar[i])
                    ret_advantages[uav_index].append(adv[i])

        return np.array(ret_states), np.array(ret_actions), np.array(ret_targets), np.array(ret_advantages)


def _main():
    ray.init()
    worker = PPOWorker.remote(PPOArgs())
    print(ray.get(worker.collect.remote(10)))
    ray.shutdown()


if __name__ == '__main__':
    _main()
