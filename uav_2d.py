import math
import time

import numpy as np
from PIL import Image

from utils import draw_uav, draw_obstacle, draw_tr, draw_target_area


class UavModel2D:

    def __init__(self, init_x=0, init_y=0, init_v=0, init_w=0):
        self.x = init_x
        self.y = init_y
        self.v = init_v
        self.w = init_w

    def step(self, control):
        assert -np.pi / 4 <= control[0] <= np.pi / 4
        assert -1.0 <= control[1] <= 1.0
        self.w += control[0]
        self.v += control[1]
        if self.v > 10:
            self.v = 10
        if self.v < 0:
            self.v = 0
        while self.w > np.pi * 2:
            self.w -= np.pi * 2
        while self.w < -np.pi * 2:
            self.w += np.pi * 2
        self.x += self.v * math.cos(self.w)
        self.y += self.v * math.sin(self.w)


class ManyUavEnv:

    OBSTACLE_CNT = 40
    COLLISION_R = 30
    TARGET_R = 100

    def __init__(self, uav_cnt, train_mode):
        self._uav_cnt = uav_cnt
        self._viewer = None

        self.uav_s = [UavModel2D(init_x=np.random.uniform(0, 2000), init_y=np.random.uniform(0, 400), init_w=np.pi / 2)
                      for _ in range(uav_cnt)]

        self.obstacle = [(np.random.uniform(0, 2000), np.random.uniform(500, 1500))
                         for _ in range(ManyUavEnv.OBSTACLE_CNT)]
        self.obstacle_collision = [False for _ in range(ManyUavEnv.OBSTACLE_CNT)]

        self.trace = [[] for _ in range(uav_cnt)]

        self.train_mode = train_mode
        self.target = (1000, 1750)
        self.step_cnt = 0

    def get_observation(self):
        observation = []
        for u in self.uav_s:
            rotate_mat = np.array([
                [np.cos(u.w), -np.sin(u.w)],
                [np.sin(u.w), np.cos(u.w)]
            ])
            pos = np.array([u.x, u.y])
            uav_dir = rotate_mat @ np.array([1., 0.])
            uav_right_dir = np.array([uav_dir[1], -uav_dir[0]])
            # obstacle info
            obstacle_dist = [2000. for _ in range(12)]
            for obs in self.obstacle:
                obs_dir = (np.array([obs[0], obs[1]]) - pos)
                obs_dir = obs_dir / (obs_dir @ obs_dir) ** .5
                if np.cross(obs_dir, uav_right_dir) < 0:
                    # print(f'dot: {np.dot(obs_dir, uav_right_dir)}')
                    # print(f'uav: {uav_right_dir}')
                    c = math.acos(np.dot(obs_dir, uav_right_dir))

                    index = int(math.floor(12 * c / np.pi))
                    if index == 12:
                        index -= 1
                    # print(index)
                    dist = np.linalg.norm(np.array([obs[0], obs[1]]) - pos)
                    if dist < obstacle_dist[index]:
                        obstacle_dist[index] = dist
            # self info
            gps = [(u.x - 1000.) / 2000., (u.y - 1000.) / 2000., (u.v - 5.) / 10., 0.5 * u.w / np.pi]
            observation.append(
                np.hstack((gps, (np.array(obstacle_dist) - 1000) / 2000.))
            )
        return np.array(observation)[0]  # TODO: fix it

    def get_reward(self, prev_pos, next_pos):
        result = [-3] * self._uav_cnt
        self.obstacle_collision = [False for _ in range(ManyUavEnv.OBSTACLE_CNT)]
        for u in range(self._uav_cnt):
            dist_prev = np.linalg.norm(prev_pos[u] - self.target)
            dist_next = np.linalg.norm(next_pos[u] - self.target)

            # reduce distance, with velocity penalty
            result[u] += math.tanh(0.2 * (10 - self.uav_s[u].v)) * (dist_prev - dist_next)

            collision = False
            for i, o in enumerate(self.obstacle):
                if np.linalg.norm(next_pos[u] - o) < ManyUavEnv.COLLISION_R:
                    collision = True
                    self.obstacle_collision[i] = True
                    break
            if collision:
                result[u] -= 20                  # no collision

            if np.linalg.norm(next_pos[u] - self.target) < ManyUavEnv.TARGET_R:  # formation in the circle
                result[u] += 5
        return result[0]  # TODO: fix it

    def reset(self):
        self.uav_s = [UavModel2D(init_x=np.random.uniform(0, 2000), init_y=np.random.uniform(0, 400))
                      for _ in range(self._uav_cnt)]
        self.obstacle = [(np.random.uniform(0, 2000), np.random.uniform(500, 1500))
                         for _ in range(ManyUavEnv.OBSTACLE_CNT)]
        self.obstacle_collision = [False for _ in range(ManyUavEnv.OBSTACLE_CNT)]
        self.trace = [[] for _ in range(self._uav_cnt)]
        self.step_cnt = 0
        return self.get_observation()

    def step(self, actions):
        prev_pos = []
        next_pos = []
        for i, u in enumerate(self.uav_s):
            prev_pos.append(np.array([u.x, u.y]))
            u.step(actions)  # TODO: fix it
            next_pos.append(np.array([u.x, u.y]))
        if not self.train_mode:
            for i, u in enumerate(self.uav_s):
                self.trace[i].append((u.x, u.y))
        self.step_cnt += 1
        return self.get_observation(), self.get_reward(prev_pos, next_pos), self.step_cnt > 500, {}

    def render(self, mode='human'):
        image = Image.new(mode='RGB', size=(800, 800), color='white')
        transform = np.array([
            [800 / 2000, 0],
            [0, 800 / 2000]
        ])

        draw_target_area(image, (1000, 1750), transform, ManyUavEnv.TARGET_R)

        for uav in self.uav_s:
            position = np.array([uav.x, uav.y])
            draw_uav(image, position, transform)
        for i, o in enumerate(self.obstacle):
            position = np.array([o[0], o[1]])
            draw_obstacle(image, position, transform, self.obstacle_collision[i], ManyUavEnv.COLLISION_R)

        draw_tr(image, self.trace, transform)
        image = np.asarray(image)
        if mode == 'rgb_array':
            return image
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self._viewer is None:
                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(image)
            return self._viewer.isopen

    def close(self):
        if self._viewer:
            self._viewer.close()

