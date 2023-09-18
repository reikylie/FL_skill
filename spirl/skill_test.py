from collections import OrderedDict
import os
from typing import Dict, List, Optional, Tuple, Union
import matplotlib; matplotlib.use('Agg')
import torch
from shutil import copy
import numpy as np
import Custome_train3
from spirl.components.params import get_args
from spirl.utils.general_utils import AttrDict
import random

def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seeds(seed=0)
init_data_dir = "./data7/kitchen/"
config = AttrDict(
    path = "spirl/configs/skill_prior_learning/kitchen/FL_clip",
    prefix = "normal-learning with tasks-4",
    new_dir = False,
    dont_save = False,
    resume = "",
    train = True,
    test_prediction = True,
    skip_first_val = False,
    val_sweep = False,
    gpu = 0,
    strict_weight_loading = True,
    deterministic = False,
    log_interval = 500,
    per_epoch_img_logs = 1,
    val_data_size = 160,
    val_interval = 5,
    detect_anomaly = False,
    feed_random_data = False,
    train_loop_pdb = False,
    debug = False,
    save2mp4 = False
)
init_model = Custome_train3.ModelTrainer(args=config,cid=0, data_dir = init_data_dir)
