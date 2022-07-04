import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.modules.loss
from utils.data_utils import normalize_adj,sparse_mx_to_torch_sparse_tensor
torch.set_printoptions(threshold=np.inf) 


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join(
            ["{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items() if metric_name != 'log_softmax_p'])


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """
    Adds a flag (and default value) to an ArgumentParser for each parameter in a config
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser

def adj_gen(args, dgl_graph_ST, g, toAdd_list):
    src, dst = toAdd_list[0], toAdd_list[1]
    adj_list, adj4dgl_list = [], []
    edge_ids = g.edge_ids(src, dst)
    #edge_degrees = g.edata['edge_degree'][edge_ids]
    similarities = g.edata['similarity'][edge_ids]
    coef = similarities / similarities.mean()
    #coef = edge_degrees / edge_degrees.mean()
    
    add_rate = args.add_rate
    drop_rates = torch.FloatTensor(np.ones(len(src)) * add_rate)
    #drop_rates = drop_rates * coef
    #drop_rates = torch.clamp(drop_rates, 0, 0.95)
    
    for i in range(args.num_adj):
        tmp = dgl_graph_ST.local_var()
        add_src, add_dst = [], []
        flags = torch.bernoulli(drop_rates)
        for i in range(len(flags)):
            if(flags[i]):
                add_src.append(src[i])
                add_dst.append(dst[i])
        tmp.add_edges(add_src, add_dst)
        adj = tmp.adj(scipy_fmt='csr')
        adj4dgl = normalize_adj(adj)
        #adj = sparse_mx_to_torch_sparse_tensor(adj4dgl).to(args.device)
        #adj_list.append(adj)
        adj4dgl_list.append(adj4dgl)

    
    return  adj4dgl_list
        
