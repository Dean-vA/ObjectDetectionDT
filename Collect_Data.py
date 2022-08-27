#%%
import gym
import numpy as np
from PIL import Image
import os
from pyglet.window import key

from controller import PurePursuitPolicy
from gym_duckietown.envs import DuckietownEnv

#make directory if it doesn´t exist
if not os.path.exists('./data'):
    os.makedirs('./data')

MAX_FRAMES = 1000

#env = gym.make("Duckietown-loop_obstacles-v0")
env = DuckietownEnv(
    domain_rand=True, max_steps=MAX_FRAMES, randomize_maps_on_reset=False, map_name="loop_obstacles"
)
obs = env.reset()
env.render()

policy = PurePursuitPolicy(env)

for frame in range(MAX_FRAMES):
    action = list(policy.predict(np.array(obs)))
    action[1] *= 7

    obs, reward, done, info = env.step(action)
    env.render()

    im = Image.fromarray(obs)
    im.save("data/" + str(frame) + ".jpeg")

#seg = env.render(segment=True)
#seg = env.render_obs(True)
# %%
