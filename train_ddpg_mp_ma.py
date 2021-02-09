import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from deeprl_utils.buffer import ReplayBuffer
from uav_2d_ma_fast import ManyUavEnv
from worker_ma import WorkerManager


torch.set_num_threads(4)
CHIP = 'cpu'


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
        self.state_dim, self.action_dim = 22, 2
        self._args = args
        self._actor = pnet_cls(self.state_dim, self.action_dim).to(CHIP)
        self._critic = qnet_cls(self.state_dim, self.action_dim).to(CHIP)
        self._target_actor = pnet_cls(self.state_dim, self.action_dim).to(CHIP)
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic = qnet_cls(self.state_dim, self.action_dim).to(CHIP)
        self._target_critic.load_state_dict(self._critic.state_dict())
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=self._args.actor_lr)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=self._args.critic_lr)
        self._critic_loss_fn = torch.nn.MSELoss()
        self._exp = \
            [ReplayBuffer(args.exp_cap, (self.state_dim, ), self.action_dim) for _ in range(args.agents)]

        self._actor.share_memory()

        self._worker = WorkerManager(8, self._actor, self._args)

        self._update_cnt = 0
        self._sw = SummaryWriter(self._args.log_dir)

        self._env = ManyUavEnv(self._args.agents, 123, self._args.reward_type)

        self._critic_update_cnt = 0
        self._interact_cnt = 0

        # TODO: load prev best model and retrain
        if args.curr_model_name:
            self._init_policy(args.curr_model_name)
        
    def _init_policy(self, file_name):
        saved_data = torch.load(file_name)
        self._actor.load_state_dict(saved_data[0])
        self._target_actor.load_state_dict(saved_data[0])
#         self._critic.load_state_dict(saved_data[1])
#         self._target_critic.load_state_dict(saved_data[1])
        print(f'TestModel: {self.test_model()}')
        
        # ADD SOME DATA TO REPLAY BUFFER
        for i in range(1000):
            data = self._worker.collect()
            states, actions, rewards, states_, dones = [], [], [], [], []
            for d in data:
                for agent in range(self._args.agents):
                    states.append(d[agent][0])
                    actions.append(d[agent][1])
                    rewards.append(d[agent][2])
                    states_.append(d[agent][3])
                    dones.append(d[agent][4])
                    self._exp[agent].add(*d[agent])
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            states_ = np.array(states_)
            dones = np.array(dones)
            # use these samples to update Value function approximation
            # should update critic ?
#             loss = self._update_value_function(states, actions, rewards, states_, dones)
#             print(f'INIT POLICY VALUE {i}...')
#             self._sw.add_scalar('loss/init_value', loss, i)
#         self._target_critic.load_state_dict(self._critic.state_dict())

    def choose_action_with_exploration(self, state):
        action = self.choose_action(state)
        noise = np.random.normal(0, self._args.scale, (self.action_dim, ))
        action = np.clip(action + noise, -1, 1)  # clip action between [-1, 1]
        return action

    def choose_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(CHIP)
            action = self._actor(state)
        action = action.detach().cpu().numpy()
        return action

    def soft_copy_parms(self):
        with torch.no_grad():
            for t, s in zip(self._target_actor.parameters(), self._actor.parameters()):
                t.copy_(self._args.tau * t.data + (1 - self._args.tau) * s.data)
            for t, s in zip(self._target_critic.parameters(), self._critic.parameters()):
                t.copy_(self._args.tau * t.data + (1 - self._args.tau) * s.data)

    def update(self, index):
        self._update_cnt += 1
        
        samples = self._exp[index].sample(self._args.batch)
        s, a, r, s_, d = samples
        s_feed = torch.tensor(s).float().to(CHIP)
        a_feed = torch.tensor(a).float().to(CHIP)
        r_feed = torch.tensor(r).float().to(CHIP)
        next_s_feed = torch.tensor(s_).float().to(CHIP)
        
        # update critic
        with torch.no_grad():
            opt_a = self._target_actor(next_s_feed)
            target_critic_input = torch.cat((next_s_feed, opt_a), 1)
            target_critic_output = self._target_critic(target_critic_input)
            target_critic_output[d] = 0
            target_critic_output *= self._args.gamma
            target_critic_output += r_feed.view(-1, 1)

        critic_output = self._critic(torch.cat((s_feed, a_feed), 1))
        critic_loss = self._critic_loss_fn(critic_output, target_critic_output)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        # finished      
        self._critic_update_cnt += 1

        # update actor, maximize Q((s, actor_output(s))
        opt_a = self._actor(s_feed)
        q_input = torch.cat((s_feed, opt_a), 1)
        q_val = self._target_critic(q_input)
        actor_loss = -q_val.mean()
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()
        # finished
        
        # copy parms to target
        self.soft_copy_parms()

        # log loss
        if self._update_cnt % 5000 == 0:
            self._sw.add_scalar('loss/critic', critic_loss.detach().cpu().item(), self._update_cnt)
            self._sw.add_scalar('loss/actor', actor_loss.detach().cpu().item(), self._update_cnt)

    def train(self):
        collect_time_prev = time.time()
        data = self._worker.collect()
        collect_time_next = time.time()
        self._interact_cnt += self._args.update_interval
        
        print(f'NowAgents: {self._args.agents}')
        # print(f'collect time: {collect_time_next - collect_time_prev} s')

        add_time_prev = time.time()
        for d in data:
            for i in range(self._args.agents):
                self._exp[i].add(*d[i])
        add_time_next = time.time()

        # print(f'add time: {add_time_next - add_time_prev}')

        update_time_prev = time.time()
        for u in range(self._args.update_cnt):
            now_update = np.random.randint(0, self._args.agents)
            self.update(now_update)
        update_time_next = time.time()
        # print(f'update time: {update_time_next - update_time_prev}')

    def test_one_episode(self, viewer=False):
        state = self._env.reset()
        if viewer:
            self._env.render()
            time.sleep(0.1)
        done = False
        total_reward = 0
        while not done:
            actions = []
            for i in range(self._args.agents):
                action = self.choose_action(state[i])
                actions.append(action)
            state_, reward, done, _ = self._env.step(np.array(actions) * self._args.action_bound)
            if viewer:
                self._env.render()
                time.sleep(0.1)
            state = state_
            total_reward += np.mean(reward)   # average reward
        return total_reward

    def test_model(self, cnt=10):
        r = [self.test_one_episode() for _ in range(cnt)]
        r_mean = np.mean(r)
        self._sw.add_scalar('step_reward/test', r_mean, self._interact_cnt)
        return r_mean

    def save(self, path):
        path = os.path.join(path, f'{self._worker.now_episode()}-{self._update_cnt}.pkl')
        torch.save((self._actor.state_dict(), self._critic.state_dict()), path)

    def load(self, path, ep, cnt):
        path = os.path.join(path, f'{ep}-{cnt}.pkl')
        state_dict = torch.load(path)
        self._actor.load_state_dict(state_dict)


class DDPGArgs:

    def __init__(self):
        self.reward_type = 0
        self.exp_cap = 1000000
        self.tau = 0.95
        self.gamma = 0.95
        self.batch = 128
        self.test_interval = 32
        self.update_cnt = 200
        self.update_interval = 500
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3
        self.curr_model_name = './005/UAV_CURR_20/model/43524-11500200.pkl'
        self.env_name = 'UAV_CURR_40_PLUS'  # 注意，INIT版本要同时注释load模型部分
        self.action_bound = [np.pi / 4, 1.0]
        self.max_ep = 100000
        self.scale = 0.1
        self.agents = 40

        self.log_dir = f'./005/{self.env_name}'
        self.save_dir = f'./005/{self.env_name}/model'


def main():
    args = DDPGArgs()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    agent = DDPGAgent(DDPGQNet, DDPGActor, args)
    for u in range(200000):
        agent.train()
        if u % 500 == 0:
            agent.save(args.save_dir)
            agent.test_model(5)


if __name__ == '__main__':
    main()
