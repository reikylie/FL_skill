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
        self._env.init_qpos = np.array([-1.2150208e+00, -1.7728019e+00,  1.8986284e+00, -2.0606890e+00,
       -4.2002100e-01,  1.3374336e+00,  2.3152425e+00,  2.7378878e-02,
        2.5079377e-02,  5.8420639e-05,  9.7897602e-05, -1.2832818e-06,
        4.3017291e-05, -1.4175036e-05, -4.9132534e-05,  5.2295291e-05,
       -1.6413776e-05, -2.7060023e-04,  1.8146726e-04, -4.3322358e-04,
       -4.7870786e-03, -7.6087657e-03, -6.9890708e-01, -2.2289988e-01,
        5.7098156e-01,  1.6762449e+00,  9.7355604e-01,  1.1408262e-01,
        4.6889331e-02, -1.6854040e-01])
        self._env.TASK_ELEMENTS = ['bottom burner']


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
