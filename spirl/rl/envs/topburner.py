import numpy as np
from collections import defaultdict
import d4rl

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class KitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']\
    
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp.name)
        self._env.init_qpos = np.array([-1.1859719e+00, -1.1959008e+00,  1.4937971e+00, -2.3043234e+00,
        3.3233479e-01,  1.6422106e+00,  1.9930224e+00, -4.4310484e-03,
        3.2830045e-02, -1.1992746e-04, -3.9161413e-04, -7.9655397e-01,
       -4.5248806e-03,  1.2208996e-05,  1.4015675e-05, -7.4952845e-06,
        2.8450329e-05,  3.1551515e-04,  2.2501154e-05,  4.6929909e-04,
       -6.6485293e-03, -1.5214891e-03, -7.4345136e-01, -2.6946035e-01,
        3.4993225e-01,  1.6190597e+00,  9.9773061e-01,  8.3500650e-03,
        9.8998973e-04, -1.6257125e-04])
        self._env.TASK_ELEMENTS = ['top burner']
        
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
