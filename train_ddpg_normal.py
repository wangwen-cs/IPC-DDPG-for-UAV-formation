import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from ddpg_replay_buffer import DDPGReplayBuffer
from ddpg_network import DDPGActor, DDPGQNet
# from toy_env import ToyEnv
from env import ManyUavEnv


STATE_DIM = 22
ACTION_DIM = 2

torch.set_num_threads(1)


class NormalArgs:

    def __init__(self, eval_=False):
        self.scale = 0.1
        self.action_bound = np.array([np.pi / 4, 1.])
        self.tau = 0.95
        self.batch = 64
        self.gamma = 0.99
        self.actor_lr = 1e-3
        self.critic_lr = 1e-3
        self.n_agents = 2
        self.eval_ = eval_
        self.save_dir = None
        self.log_dir = None


class DDPGNormal:

    def __init__(self, args: NormalArgs):
        self._env = ManyUavEnv(args.n_agents)
        self._actor = DDPGActor(STATE_DIM, ACTION_DIM)
        self._critic = DDPGQNet(STATE_DIM, ACTION_DIM)
        self._target_actor = DDPGActor(STATE_DIM, ACTION_DIM)
        self._target_critic = DDPGQNet(STATE_DIM, ACTION_DIM)
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic.load_state_dict(self._critic.state_dict())
        self._actor_optim = torch.optim.Adam(self._actor.parameters(), lr=args.actor_lr)
        self._critic_optim = torch.optim.Adam(self._critic.parameters(), lr=args.critic_lr)
        self._buffer = [
            DDPGReplayBuffer(1000000, STATE_DIM, ACTION_DIM) for _ in range(args.n_agents)
        ]
        self._args = args

        if not self._args.eval_:
            self._sw = SummaryWriter(args.log_dir)
            self._update_cnt = 0
            self._interaction_cnt = 0

    def pre_fill_replay_buffer(self, n_samples):
        cnt = 0
        while True:
            states = self._env.reset()
            done = False
            while not done:
                actions = []
                for i in range(self._args.n_agents):
                    a = self._choose_action_with_exploration(states[i])
                    actions.append(a)
                next_states, rewards, done, _ = self._env.step(actions * self._args.action_bound)
                for i in range(self._args.n_agents):
                    self._buffer[i].add(states[i], actions[i], rewards[i], next_states[i], False)  # TODO: fix it
                states = next_states
                cnt += 1
                if cnt >= n_samples:
                    return

    def _choose_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float()
            action = self._actor(state)
        action = action.detach().cpu().numpy()
        return action

    def _choose_action_with_exploration(self, state):
        action = self._choose_action(state)
        noise = np.random.normal(0, self._args.scale, action.shape)
        action = np.clip(action + noise, -1, 1)  # clip action between [-1, 1]
        return action

    def _update(self, update_index):
        self._update_cnt += 1
        s, a, r, s_, d = self._buffer[update_index].sample(self._args.batch)
        s_feed = torch.tensor(s).float()
        a_feed = torch.tensor(a).float()
        r_feed = torch.tensor(r).float()
        next_s_feed = torch.tensor(s_).float()
        d_tensor = torch.tensor(d).bool()

        # update critic
        with torch.no_grad():
            opt_a = self._target_actor(next_s_feed)
            target_critic_input = torch.cat((next_s_feed, opt_a), 1)
            target_critic_output = self._target_critic(target_critic_input)
            target_critic_output[d_tensor] = 0
            target_critic_output *= self._args.gamma
            target_critic_output += r_feed.view(-1, 1)

        critic_loss_fn = torch.nn.MSELoss()
        critic_output = self._critic(torch.cat((s_feed, a_feed), 1))
        critic_loss = critic_loss_fn(critic_output, target_critic_output)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        self._critic_optim.step()
        # finished

        # update actor, maximize Q((s, actor_output(s))
        opt_a = self._actor(s_feed)
        q_input = torch.cat((s_feed, opt_a), 1)
        q_val = self._target_critic(q_input)
        actor_loss = -q_val.mean()
        self._actor_optim.zero_grad()
        actor_loss.backward()
        self._actor_optim.step()
        # finished

        if self._update_cnt % 200 == 0:
            self._sw.add_scalar('loss/critic', critic_loss.detach().cpu().item(), self._update_cnt)
            self._sw.add_scalar('loss/actor', actor_loss.detach().cpu().item(), self._update_cnt)

        # copy parms to target
        self._soft_copy_parameters()

    def _soft_copy_parameters(self):
        with torch.no_grad():
            for t, s in zip(self._target_actor.parameters(), self._actor.parameters()):
                t.copy_(self._args.tau * t.data + (1 - self._args.tau) * s.data)
            for t, s in zip(self._target_critic.parameters(), self._critic.parameters()):
                t.copy_(self._args.tau * t.data + (1 - self._args.tau) * s.data)

    def train_one_episode(self):
        states = self._env.reset()
        done = False
        total = 0
        while not done:
            actions = []
            for i in range(self._args.n_agents):
                a = self._choose_action_with_exploration(states[i])
                actions.append(a)
            next_states, rewards, done, _ = self._env.step(actions * self._args.action_bound)
            self._interaction_cnt += 1
            for i in range(self._args.n_agents):
                self._buffer[i].add(states[i], actions[i], rewards[i], next_states[i], False)  # TODO: fix it

            update_index = np.random.randint(0, self._args.n_agents)
            self._update(update_index)
            states = next_states
            total += np.mean(rewards)
        return total

    def eval_(self, n, view=False):
        res = np.array([self._eval_one_ep(view) for _ in range(n)])
        mean = np.mean(res, axis=0)
        if not self._args.eval_:
            self._sw.add_scalar('eval/average_return', mean[0], self._interaction_cnt)
            self._sw.add_scalar('eval/average_collision_with_obstacle', mean[1], self._interaction_cnt)
            self._sw.add_scalar('eval/average_collision_with_UAV', mean[2], self._interaction_cnt)
        return mean

    def _eval_one_ep(self, view):
        states = self._env.reset()
        done = False
        total = 0
        while not done:
            actions = []
            for i in range(self._args.n_agents):
                a = self._choose_action(states[i])
                actions.append(a)
            next_states, rewards, done, _ = self._env.step(actions * self._args.action_bound)
            if view:
                self._env.render()
                time.sleep(1 / 60.)
            states = next_states
            total += np.mean(rewards)
        collision_with_obstacles, collision_with_uavs = self._env.get_collision_cnt()
        return total, collision_with_obstacles, collision_with_uavs

    def save(self, path):
        data = (self._actor.state_dict(), self._critic.state_dict())
        file_name = os.path.join(path, f'{self._interaction_cnt}.pkl')
        torch.save(data, file_name)

    def load(self, path, index):
        file_name = os.path.join(path, f'{index}.pkl')
        data = torch.load(file_name)
        self._actor.load_state_dict(data[0])
        self._target_actor.load_state_dict(data[0])
        # self._critic.load_state_dict(data[1])


def _main():
    args = NormalArgs()
    args.n_agents = 20
    args.save_dir = "TOY/MODEL/DDPG_20_CURR_1"
    args.log_dir = "TOY/LOGS/DDPG_20_CURR_1"
    args.scale = 0.1
    args.actor_lr = 1e-3
    args.critic_lr = 1e-3
    agent = DDPGNormal(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    agent.load('./TOY/MODEL/DDPG_10_CURR_4', 500500)  # LOAD MODEL
    # print('FILL REPLAY BUFFERS')
    # agent.pre_fill_replay_buffer(10000)
    # print('FILL REPLAY BUFFERS END')

    for i in range(1001):
        r = agent.train_one_episode()
        # print(f'EP: {i}, R: {r}')
        if i % 20 == 0:
            print(f'EP: {i} EVAL: {agent.eval_(10)}')
            agent.save(args.save_dir)


def _eval():
    args = NormalArgs(eval_=True)
    agent = DDPGNormal(args)
    agent.load('./TOY/20_normal/model/', 80200)
    print(agent.eval_(10, False))


if __name__ == '__main__':
    _main()
    # _eval()
