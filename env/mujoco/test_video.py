import gym
from record_video import RecordVideo

from whopper2 import Whopper
from whand import WHand
from wfinger import WFinger
from wfetch import WFetch
from wfetch_push import WFetch_Push
from whalfcheetah import Whalfcheetah


# Création et configuration de l'environnement MuJoCo
# env = gym.make('Walker2d-v2', render_mode = 'rgb_array')
# env = WFinger(render = True, render_mode = 'rgb_array')
env = WFetch_Push(render = True, render_mode = 'rgb_array')
# env = Whalfcheetah(render = True, render_mode = 'rgb_array')
# env = Whopper(render = True, render_mode = 'rgb_array')
env =RecordVideo(env, './video', video_length=1000)

# Exécuter la simulation
observation = env.reset()
for _ in range(env.max_steps):
    action = env.action_space.sample() # Ou votre propre logique d'action
    observation, reward, done, trunc,  info = env.step(action)
    # if done:
    #     break
# Fermer l'environnement
env.close()
