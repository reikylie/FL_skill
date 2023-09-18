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
        self._env.init_qpos = np.array([-2.2246664e+00, -1.4129806e+00,  1.0475178e+00, -1.9299496e+00,
        2.3793522e-01,  1.8324754e+00,  1.1345068e+00,  3.4153547e-02,
        1.6451010e-04,  5.1391480e-04, -3.3406733e-04,  5.6890520e-05,
        3.3273012e-05,  1.5545613e-05, -4.3130811e-05,  5.4660508e-05,
        8.4606927e-06, -6.3296378e-05,  1.7659320e-04,  2.6348272e-01,
        2.8637601e-03,  3.0351961e-03, -7.7633512e-01, -2.3456331e-01,
        7.2968483e-01,  1.6189239e+00,  1.0036634e+00, -6.5602884e-03,
       -9.3019847e-03, -6.6156141e-02])
        self._env.TASK_ELEMENTS = ['hinge cabinet']

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
