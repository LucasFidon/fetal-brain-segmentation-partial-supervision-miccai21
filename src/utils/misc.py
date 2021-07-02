import os
import random
import numpy as np
import nibabel as nib
import torch
from torch.nn.init import kaiming_normal_
import torch.nn.functional as F
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
from functools import partial
from nested_dict import nested_dict
from src.utils.definitions import MIN_SIZE


def random_seed(seed_value):
    """
    Set all the seed to make sure that the results are reproducible
    :param seed_value: int, seed value to use
    """
    random.seed(seed_value)  # Python
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu vars
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k, v in params.items()}
    else:
        return getattr(
            params.cuda() if torch.cuda.is_available() else params,
            dtype)()

def pad_if_needed(array_3d, min_size=MIN_SIZE, return_padding_values=False):
    """
    Pad an array with zeros if needed so that each of its dimensions
    is at least equal to MIN_SIZE.
    :param array_3d: numpy array to be padded if needed
    :return: padded array.
    """
    shape = array_3d.shape
    need_padding = np.any(shape < np.array(min_size))
    if not need_padding:
        pad_list = [(0, 0)] * 3
        if return_padding_values:
            return array_3d, np.array(pad_list)
        else:
            return array_3d
    else:
        pad_list =[]
        for dim in range(3):
            diff = min_size[dim] - shape[dim]
            if diff > 0:
                margin = diff // 2
                pad_dim = (margin, diff - margin)
                pad_list.append(pad_dim)
            else:
                pad_list.append((0, 0))
        padded_array = np.pad(
            array_3d,
            pad_list,
            'constant',
            constant_values = [(0,0), (0, 0), (0, 0)],
        )
        if return_padding_values:
            return padded_array, np.array(pad_list)
        else:
            return padded_array