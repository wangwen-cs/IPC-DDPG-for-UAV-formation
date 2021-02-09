import os

import gym
import ray
import torch
import numpy as np
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from ppo_args import PPOArgs
from ppo_model import PPOModel
from ppo_worker import PPOWorker

from uav_2d_ma_fast import ManyUavEnv


class PPOAgent:

    def __init__(self, args: PPOArgs, eval_model=False):
        self._env = ManyUavEnv(args.n_agents, 123, 0)
        self._model = PPOModel(22, 2)
        self._args = args
        self._eval_model = eval_model
        self._exploration_std = args.std_begin
        if not eval_model:
            self._optim = torch.optim.Adam(self._model.parameters(), lr=args.lr)
            self._worker = [PPOWorker.remote(args) for _ in range(args.n_worker)]

            self._sw = SummaryWriter(args.log_dir)
            self._now_ep = 0
            self._interaction_cnt = 0

    def _choose_action(self, state, deterministic=False):
        s_tensor = torch.tensor(state).float()
        with torch.no_grad():
            v, mu = self._model(s_tensor)
            if deterministic:
                action_tensor = mu
            else:
                dist = Normal(mu, self._exploration_std)
                action_tensor = dist.sample()
                action_tensor = torch.clamp(action_tensor, -1, 1, out=action_tensor)
        return action_tensor.numpy()

    def _update(self, states, actions, targets, advantages):
        states_tensor = torch.tensor(states).float()
        actions_tensor = torch.tensor(actions).float()
        targets_tensor = torch.tensor(targets).float()
        advantages_tensor = torch.tensor(advantages).float()

        with torch.no_grad():
            _, pi_old_mu = self._model(states_tensor)
            old_dist = Normal(pi_old_mu, self._exploration_std)
            old_log_prob = old_dist.log_prob(actions_tensor).sum(axis=1)  # LOG PROB

        averaged_v_loss = 0
        averaged_p_loss = 0

        value_loss_fn = torch.nn.MSELoss()
        for _ in range(self._args.n_updates):
            # optim
            v, pi_new_mu = self._model(states_tensor)
            new_dist = Normal(pi_new_mu, self._exploration_std)
            new_log_prob = new_dist.log_prob(actions_tensor).sum(axis=1)
            r_tau = torch.exp(new_log_prob - old_log_prob)

            v_loss = value_loss_fn(v.view(-1), targets_tensor.view(-1))
            p_loss_1 = r_tau.view(-1) * advantages_tensor.view(-1)
            p_loss_2 = torch.clamp(r_tau.view(-1), 1 - self._args.clip, 1 + self._args.clip) * advantages_tensor.view(-1)
            p_loss = torch.min(p_loss_1, p_loss_2).mean()
            loss = self._args.v_c * v_loss - p_loss
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            averaged_v_loss += v_loss.detach().item()
            averaged_p_loss += p_loss.detach().item()

            averaged_v_loss /= self._args.n_updates
            averaged_p_loss /= self._args.n_updates
            # optim end
        self._sw.add_scalar('loss/v_loss', averaged_v_loss, self._now_ep)
        self._sw.add_scalar('loss/p_loss', averaged_p_loss, self._now_ep)

    def train_one_epoch(self):
        parm = self._model.state_dict()

        n_agents = self._args.n_agents
        states, actions, targets, advantages = [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)]
        data_ref = []
        for w in self._worker:
            data_ref.append(w.collect.remote(self._args.samples_per_worker, parm, self._exploration_std))

        for d_ref in data_ref:
            for uav_index in range(n_agents):
                s, a, tar, adv = ray.get(d_ref)
            
                for i in range(self._args.samples_per_worker):
                    states[uav_index].append(s[uav_index][i])
                    actions[uav_index].append(a[uav_index][i])
                    targets[uav_index].append(tar[uav_index][i])
                    advantages[uav_index].append(adv[uav_index][i])

        self._now_ep += 1
        self._interaction_cnt += self._args.n_worker * self._args.samples_per_worker
        
        ni = np.random.randint(0, n_agents)
        self._update(states[ni], actions[ni], targets[ni], advantages[ni])
        
        self._exploration_std *= self._args.std_anneal
        if self._exploration_std < self._args.std_end:
            self._exploration_std = self._args.std_end
        self._sw.add_scalar('exploration/std', self._exploration_std, self._now_ep)

    def _eval(self, view):
        s = self._env.reset()
        total = 0
        while True:
            actions = []
            for ns in s:
                a = self._choose_action(ns, True)
                actions.append(a)
            s_, r, done, _ = self._env.step(actions)
            total += np.mean(r)
            if view:
                self._env.render()
            s = s_
            if done:
                break
        return total

    def eval_(self, n=10, view=False):
        res = [self._eval(view) for _ in range(n)]
        r_mean = np.mean(res)
        if not self._eval_model:
            self._sw.add_scalar('step_reward/test', r_mean, self._interaction_cnt)
        return r_mean

    def save(self, fn):
        torch.save(self._model.state_dict(), fn)

    def load(self, fn):
        state_dict = torch.load(fn)
        self._model.load_state_dict(state_dict)


def _main():
    ray.init()
    args = PPOArgs()
    agent = PPOAgent(args)
    if not os.path.exists(f'./model/{args.env_name}'):
        os.makedirs(f'./model/{args.env_name}')
    try:
        for ep in range(10000):
            agent.train_one_epoch()
            if ep % 100 == 0:
                print(f'EP: {ep} R: {agent.eval_(n=5)}')
                agent.save(f'model/{args.env_name}/{ep}.pkl')
    finally:
        print('SHUTDOWN RAY WORKERS')
        ray.shutdown()


def _eval():
    args = PPOArgs()
    agent = PPOAgent(args, eval_model=True)
    agent.load('./model/1840.pkl')
    print(agent.eval_(10, True))


if __name__ == '__main__':
    _main()
#     _eval()