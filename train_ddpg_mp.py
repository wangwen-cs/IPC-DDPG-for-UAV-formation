import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from deeprl_utils.buffer import ReplayBuffer
from uav_2d import ManyUavEnv
from worker import WorkerManager

torch.set_num_threads(4)

class DDPGActor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._fc0 = nn.Linear(state_dim, 128)
        self._fc1 = nn.Linear(128, 64)
        self._fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self._fc0(x))
        x = torch.relu(self._fc1(x))
        x = self._fc2(x)
        return torch.tanh(x)


class DDPGQNet(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._fc0 = nn.Linear(state_dim + action_dim, 256)
        self._fc1 = nn.Linear(256, 128)
        self._fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tanh(self._fc0(x))
        x = torch.tanh(self._fc1(x))
        x = self._fc2(x)
        return x


class DDPGAgent:

    def __init__(self, qnet_cls, pnet_cls, args):
        self.state_dim, self.action_dim = 16, 2
        self._args = args
        self._actor = pnet_cls(self.state_dim, self.action_dim)
        self._critic = qnet_cls(self.state_dim, self.action_dim)
        self._target_actor = pnet_cls(self.state_dim, self.action_dim)
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic = qnet_cls(self.state_dim, self.action_dim)
        self._target_critic.load_state_dict(self._critic.state_dict())
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=self._args.actor_lr)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=self._args.critic_lr)
        self._critic_loss_fn = torch.nn.MSELoss()
        self._exp = ReplayBuffer(args.exp_cap, (self.state_dim, ), self.action_dim)

        self._actor.share_memory()

        self._worker = WorkerManager(8, self._actor, self._args)

        self._update_cnt = 0
        self._sw = SummaryWriter(self._args.log_dir)

        self._env = ManyUavEnv(1, True)

    def choose_action_with_exploration(self, state):
        action = self.choose_action(state)
        noise = np.random.normal(0, self._args.scale, (self.action_dim, ))
        action = np.clip(action + noise, -1, 1)  # clip action between [-1, 1]
        return action

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action = self._actor(state)
        action = action.detach().numpy()
        return action

    def soft_copy_parms(self):
        with torch.no_grad():
            for t, s in zip(self._target_actor.parameters(), self._actor.parameters()):
                t.copy_(0.95 * t.data + 0.05 * s.data)
            for t, s in zip(self._target_critic.parameters(), self._critic.parameters()):
                t.copy_(0.95 * t.data + 0.05 * s.data)

    def update(self):
        samples = self._exp.sample(self._args.batch)
        s, a, r, s_, d = samples

        # update critic
        with torch.no_grad():
            opt_a = self._target_actor(torch.Tensor(s_)).detach().numpy()
            target_critic_input = np.hstack((s_, opt_a))
            target_critic_output = self._target_critic(torch.FloatTensor(target_critic_input))
            target_critic_output[d] = 0
            target_critic_output *= self._args.gamma
            target_critic_output += r.reshape((-1, 1))
            target_critic_output = target_critic_output.float()

        critic_output = self._critic(torch.FloatTensor(np.hstack((s, a))))
        critic_loss = self._critic_loss_fn(critic_output, target_critic_output)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        # finished

        # update actor, maximize Q((s, actor_output(s))
        opt_a = self._actor(torch.FloatTensor(s))
        q_input = torch.cat([torch.FloatTensor(s), opt_a], 1)
        q_val = self._target_critic(torch.FloatTensor(q_input))
        actor_loss = -q_val.mean()
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()
        # finished

        # copy parms to target
        self.soft_copy_parms()
        self._update_cnt += 1

        # log loss
        if self._update_cnt % 100 == 0:
            self._sw.add_scalar('loss/actor', actor_loss.detach().item(), self._update_cnt)
            self._sw.add_scalar('loss/critic', critic_loss.detach().item(), self._update_cnt)

    def train(self):
        collect_time_prev = time.time()
        data = self._worker.collect()
        collect_time_next = time.time()

        print(f'collect time: {collect_time_next - collect_time_prev} s')

        add_time_prev = time.time()
        for d in data:
            self._exp.add(d[0], d[1], d[2], d[3], d[4])
        add_time_next = time.time()

        print(f'add time: {add_time_next - add_time_prev}')

        update_time_prev = time.time()
        for u in range(self._args.update_cnt):
            self.update()
        update_time_next = time.time()
        print(f'update time: {update_time_next - update_time_prev}')

    def test_one_episode(self, viewer=False):
        state = self._env.reset()
        if viewer:
            self._env.render()
            time.sleep(0.1)
        done = False
        total_reward = 0
        while not done:
            action = self.choose_action(state)
            state_, reward, done, _ = self._env.step(action * self._args.action_bound)
            if viewer:
                self._env.render()
                time.sleep(0.1)
            state = state_
            total_reward += reward
        return total_reward

    def test_model(self, cnt=10):
        r = [self.test_one_episode() for _ in range(cnt)]
        r_mean = np.mean(r)
        self._sw.add_scalar('step_reward/test', r_mean, self._update_cnt)
        return r_mean

    def save(self, path):
        path = os.path.join(path, f'{self._worker.now_episode()}-{self._update_cnt}.pkl')
        torch.save(self._actor.state_dict(), path)

    def load(self, path, ep, cnt):
        path = os.path.join(path, f'{ep}-{cnt}.pkl')
        state_dict = torch.load(path)
        self._actor.load_state_dict(state_dict)


class DDPGArgs:

    def __init__(self):
        self.exp_cap = 1000000
        self.gamma = 0.95
        self.batch = 128
        self.test_interval = 32
        self.update_cnt = 200
        self.update_interval = 500
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        self.env_name = 'UAV_0'
        self.action_bound = [np.pi / 4, 1.0]
        self.max_ep = 100000
        self.scale = 0.1

        self.log_dir = './logs/ddpg/{}'.format(self.env_name)


def main():
    args = DDPGArgs()

    agent = DDPGAgent(DDPGQNet, DDPGActor, args)
    for u in range(200000):
        agent.train()
        if u % 50 == 0:
            agent.save('./result')
            agent.test_model(5)
            

if __name__ == '__main__':
    main()
