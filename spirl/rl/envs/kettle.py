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
        self._env.init_qpos = np.array([-7.4123758e-01, -1.7546432e+00,  1.9558041e+00, -1.6154059e+00,
       -5.4668921e-01,  1.2045333e+00,  2.4618077e+00,  2.5471697e-02,
        1.9734293e-02, -7.9886944e-05, -4.2385687e-04,  1.9689563e-05,
        3.9980623e-05,  6.9879199e-05,  2.0673247e-05, -1.9986381e-05,
        2.8734530e-05, -3.9337866e-04,  2.5174528e-04,  1.2424939e-05,
        5.7478216e-03,  5.6894189e-03, -5.8231127e-01, -2.6878706e-01,
        3.4955165e-01,  1.6189430e+00,  1.0019971e+00, -7.0487643e-03,
       -2.1925075e-03, -4.1788328e-04])
        self._env.TASK_ELEMENTS = ['kettle']
    
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
