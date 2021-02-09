import numpy as np
from PIL import Image

from utils import *
from cppenv import env as cpp_env


class ManyUavEnv:

    TARGET_R = 100
    COLLISION_R = 30

    def __init__(self, uav_cnt, seed, reward_type):
#         print(f'Env SEED: {seed}')
        self._cpp_env = cpp_env.ManyUavEnv(uav_cnt, seed, reward_type)
        self._viewer = None

    def reset(self):
        self._cpp_env.reset()
        obs = self._cpp_env.getObservations()
        return np.array(obs)

    def step(self, actions):
        self._cpp_env.step(cpp_env.ControlInfo(actions))
        obs = np.array(self._cpp_env.getObservations())
        rewards = np.array(self._cpp_env.getRewards())
        done = self._cpp_env.isDone()
        return obs, rewards, done, {}

    def render(self, mode='human'):
        image = Image.new(mode='RGB', size=(800, 800), color='white')
        transform = np.array([
            [800 / 2000, 0],
            [0, 800 / 2000]
        ])

        target_pos = self._cpp_env.getTarget()
        draw_target_area(image, (target_pos.x, target_pos.y), transform, ManyUavEnv.TARGET_R)

        uavs = self._cpp_env.getUavs()
        for u in uavs:
            draw_uav(image, [u.x, u.y], transform)

        obs = self._cpp_env.getObstacles()
        collision = self._cpp_env.getCollision()
        for i, o in enumerate(obs):
            draw_obstacle(image, [o.x, o.y], transform, collision[i], ManyUavEnv.COLLISION_R)

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


def main():
    env = ManyUavEnv(4, True)
    env.reset()
    done = False
    while not done:
        s, r, done, info = env.step(np.random.uniform(-1, 1, (4, 2)))
        env.render()
    env.close()


if __name__ == '__main__':
    main()
