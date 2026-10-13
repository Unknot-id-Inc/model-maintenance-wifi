import os
import os.path as osp
import pickle
import json
import numpy as np
from numpy.linalg import norm
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import argparse
import shutil
import tqdm
import random
from torch import Tensor, transpose, matmul, sigmoid, diag
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_geometric.utils import from_networkx, negative_sampling
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

def load_file_comparison(filename, observation_ids, observation_sign, mac_all, file_type, prune):
    raise RuntimeError(
        "The legacy list-based parser was removed because it truncated/handled "
        "BSSIDs differently from prep.py. Use gufu_data.prepare_dataset_files "
        "and consume its PreparedDataset; old list output cannot carry masks or provenance."
    )

def find_indices(cur_list, item):
    indices = []
    for idx, value in enumerate(cur_list):
        if value == item:
            indices.append(idx)
    return indices

def series_to_file(obj, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(obj, f_out, -1)
        print("Data written into {}".format(filename))


def file_to_series(filename):
    with open(filename, 'rb') as f_in:
        series = pickle.load(f_in)
        print("File {} loaded.".format(filename))
        return series


def rssi2weight(offset, rssi):
    return offset + rssi


def is_virtual_mac(mac_addr):
    mac_addr = mac_addr.replace(":", "").upper()
    first_hex = int(mac_addr[0:2], 16)
    return first_hex & 0x02 != 0

def get_data(prepared_dataset, encoder_features, *, config=None):
    """Build the shared, ID-aligned graph state from prepared and learned data.

    The former edgelist/default-zero-Z interface is deliberately removed: it
    assigned rows using set iteration and fabricated random sample features.
    """
    from graph_build import GraphBuildConfig, build_graph_state
    from gufu_data import PreparedDataset
    from ae import EncoderFeatureArtifact

    if not isinstance(prepared_dataset, PreparedDataset):
        raise TypeError("get_data requires a PreparedDataset; edgelist input is no longer supported")
    if not isinstance(encoder_features, EncoderFeatureArtifact):
        raise TypeError("get_data requires an ID-aligned EncoderFeatureArtifact")
    if config is not None and not isinstance(config, GraphBuildConfig):
        raise TypeError("config must be a GraphBuildConfig")
    return build_graph_state(prepared_dataset, encoder_features, config)

def interpolate_point(timestamp, timestamps, breakpoints):
    if timestamp <= timestamps[0]:
        print("timestamp too small: {} <= {}".format(timestamp, timestamps[0]))
        return breakpoints[0]
    if timestamp >= timestamps[-1]:
        print("timestamp too large: {} >= {}".format(
            timestamp, timestamps[-1]))
        return breakpoints[-1]

    for idx in range(len(timestamps)-1):
        if timestamps[idx] <= timestamp <= timestamps[idx+1]:
            return [breakpoints[idx][coor_id] + (timestamp - timestamps[idx]) /
                    (timestamps[idx+1] - timestamps[idx]) *
                    (breakpoints[idx+1][coor_id] - breakpoints[idx][coor_id])
                    for coor_id in [0, 1]]    
