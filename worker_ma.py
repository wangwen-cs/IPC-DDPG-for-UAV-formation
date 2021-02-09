import time

import torch
from torch.multiprocessing import Process, Queue, Event, Value, set_start_method
import numpy as np
from uav_2d_ma_fast import ManyUavEnv

# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

CHIP = 'cpu'


class Worker:

    def __init__(self, queue: Queue, collect_event: Event, actor_net, args, seed):
        self._queue = queue
        self._collect_event = collect_event
        self._actor = actor_net
        self._args = args
        self.event = Event()
        self.seed = seed
        
    def run(self, episode):
        env = ManyUavEnv(self._args.agents, self.seed, self._args.reward_type)
        state = env.reset()
        while True:
            self.event.set()
            self._collect_event.wait()
            actions = []
            for i in range(self._args.agents):
                action = self._choose_action_with_exploration(state[i])
                actions.append(action)
            next_state, reward, done, info = env.step(np.array(actions) * self._args.action_bound)

            transition = []
            for i in range(self._args.agents):
                transition.append((state[i], actions[i], reward[i], next_state[i], done))

            self._queue.put(transition)

            state = next_state
            if done:
                state = env.reset()
                with episode.get_lock():
                    episode.value += 1
            if self._queue.qsize() >= self._args.update_interval:
                self._collect_event.clear()

    def _choose_action_with_exploration(self, state):
        action = self._choose_action(state)
        noise = np.random.normal(0, self._args.scale, (2, ))
        action = np.clip(action + noise, -1, 1)  # clip action between [-1, 1]
        return action

    def _choose_action(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().to(CHIP)
            action = self._actor(state)
        action = action.detach().cpu().numpy()
        return action


class WorkerManager:

    def __init__(self, n_workers, actor, args):
        self._now_episode = Value('i', 0)

        self.queue = Queue()
        self.collect_event = Event()

        self.worker = []
        for i in range(n_workers):
            self.worker.append(Worker(self.queue, self.collect_event, actor, args, i))
        self.process = [Process(target=self.worker[i].run, args=(self._now_episode, )) for i in range(n_workers)]

        for p in self.process:
            p.start()
        print(f'Start {n_workers} workers.')

    def collect(self):
        result = []
        self.collect_event.set()
        while self.collect_event.is_set():
            # WAIT FOR DATA COLLECT END
            pass

        for w in self.worker:
            w.event.wait()

        while not self.queue.empty():
            result.append(self.queue.get())

        for w in self.worker:
            w.event.clear()
        return result

    def now_episode(self):
        value = self._now_episode.value
        return value
