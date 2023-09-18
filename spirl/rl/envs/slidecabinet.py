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
        self._env.init_qpos = np.array([-1.2674966e+00, -1.5856974e+00,  1.4021021e+00, -1.9417316e+00,
        3.2742313e-01,  1.5248332e+00,  1.3944041e+00,  4.6823174e-02,
       -4.9968548e-03, -3.4511494e-04,  2.8893608e-04,  4.6096657e-06,
       -1.0731519e-06,  6.9477443e-05,  3.1747608e-05, -8.7703949e-01,
       -5.0776331e-03, -5.5758768e-01, -4.0247560e-02,  3.0837418e-04,
        5.3328201e-03,  6.5747132e-03,  3.2931585e-03, -2.1026017e-01,
        7.5243098e-01,  1.6189126e+00,  1.0055639e+00,  1.8677633e-03,
       -2.8992987e-03,  8.8044100e-02])
        self._env.TASK_ELEMENTS = ['slide cabinet']
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
