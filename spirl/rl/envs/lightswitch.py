import numpy as np
from collections import defaultdict
import d4rl

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class KitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']
    
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self._env.init_qpos = np.array([-1.2935489e+00, -1.2578226e+00,  1.3557991e+00, -2.2373316e+00,
        9.9678762e-02,  1.8058912e+00,  2.1831472e+00,  1.9958969e-03,
        3.7682928e-02,  6.9472124e-05,  1.5574187e-04,  6.4468877e-05,
       -1.9493506e-05, -1.1361010e-05,  6.3432879e-08, -8.1335491e-01,
       -4.6323389e-03, -4.3606698e-05, -6.2289029e-05,  3.9835775e-04,
        4.9169930e-03,  8.3130744e-04, -7.8612494e-01, -2.6859683e-01,
        3.5014573e-01,  1.6188477e+00,  9.9919045e-01, -5.6040259e-03,
       -1.2207832e-04, -4.4425117e-04])
        self._env.TASK_ELEMENTS = ['light switch'] 
        
    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "kitchen-mixed-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        completed_subtasks = info.pop("completed_tasks")
        for task in self.SUBTASKS:
            self.solved_subtasks[task] = 1 if task in completed_subtasks or self.solved_subtasks[task] else 0
        return info


class NoGoalKitchenEnv(KitchenEnv):
    """Splits off goal from obs."""
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        obs = obs[:int(obs.shape[0]/2)]
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:int(obs.shape[0]/2)]
