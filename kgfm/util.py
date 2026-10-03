import os
import ast
import copy
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch_geometric.utils.sparse import index2ptr
from torch_geometric.utils import index_sort

from kgfm import datasets 

import numpy as np


logger = logging.getLogger(__file__)


def trainable_parameters(model):
    """Return a list of parameters with requires_grad=True."""
    return [p for p in model.parameters() if p.requires_grad]


# Maps cfg.model["class"] to the per-structure cache the dataset must load.
# The cache key is also the filename suffix (data_<key>.pt) and the lookup key
# in datasets.BUILDERS. For TRIX both classes share the same on-disk file,
# even though the file is named "_entity"; the contents are still data.relation_adj.
RELATION_CACHE_KEY_BY_MODEL = {
    "Ultra":         "relation_graph",
    "MOTIF":         "relation_hypergraph",
    "TRIXEntity":    "relation_graph_entity",
    "TRIXRelation":  "relation_graph_entity",
    # Flock walks the base graph at forward time; MERIT pretraining rebuilds its motif
    # tensor per batch. Both only need the standard cache.
    "Flock":         "relation_graph",
    "MERIT":         "relation_graph",
}

# At evaluation on a single dataset MERIT reads its motif tensor from the "motif" cache.
MOTIF_CACHE_MODELS = {"MERIT"}


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    tree = env.parse(raw)
    vars = meta.find_undeclared_variables(tree)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return resolve_config_paths(cfg)


REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _is_null(value):
    return value is None or str(value).strip().lower() in ("", "none", "null")


def repo_path(path):
    """`~` expands; a relative path resolves against the repository root, so configs
    and command-line paths do not depend on the working directory."""
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, path)
    return os.path.normpath(path)


def resolve_config_paths(cfg):
    """Make output_dir, dataset.root and checkpoint paths absolute, in place."""
    if not _is_null(cfg.get("output_dir")):
        cfg.output_dir = repo_path(cfg.output_dir)
    dataset = cfg.get("dataset")
    if dataset is not None and not _is_null(dataset.get("root")):
        dataset.root = repo_path(dataset.root)
    for key in ("checkpoint", "backbone_ckpt", "query_ckpt"):
        if key in cfg:
            cfg[key] = None if _is_null(cfg[key]) else repo_path(cfg[key])
    return cfg


def literal_eval(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
    
    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, required=True)
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars


def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def get_device(cfg):
    if cfg.train.gpus:
        device = torch.device(cfg.train.gpus[get_rank()])
    else:
        device = torch.device("cpu")
    return device


def create_working_directory(cfg):
    # Per-job filename so concurrent run_many.py invocations sharing the same
    # output_dir don't race on a single `working_dir.tmp`. All ranks of a given
    # SLURM job share the same SLURM_JOB_ID, so the file still works as the
    # rank-0 → other-ranks sync mechanism it was built for.
    file_name = f"working_dir.{os.environ.get('SLURM_JOB_ID', os.getpid())}.tmp"
    world_size = get_world_size()
    if cfg.train.gpus is not None and len(cfg.train.gpus) != world_size:
        error_msg = "World size is %d but found %d GPUs in the argument"
        if world_size == 1:
            error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
        raise ValueError(error_msg % (world_size, len(cfg.train.gpus)))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    if cfg.dataset.get("version") is None:
        working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], time.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.model["class"], cfg.dataset["class"], str(cfg.dataset["version"]), time.strftime("%Y-%m-%d-%H-%M-%S"))
    
    # synchronize working directory
    if get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    synchronize()
    if get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    synchronize()
    if get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir



def build_dataset(cfg, device = "cpu", cache_key_override = None):
    data_config = copy.deepcopy(cfg.dataset)
    cls = data_config.pop("class")

    if cache_key_override is not None:
        cache_key = cache_key_override
    else:
        model_cls = cfg.model["class"]
        if model_cls not in RELATION_CACHE_KEY_BY_MODEL:
            raise KeyError(
                f"unknown model class {model_cls!r}; "
                f"add it to RELATION_CACHE_KEY_BY_MODEL in kgfm/util.py"
            )
        if model_cls in MOTIF_CACHE_MODELS and cls != "JointDataset":
            cache_key = "motif"
        else:
            cache_key = RELATION_CACHE_KEY_BY_MODEL[model_cls]

    ds_cls = getattr(datasets, cls)
    dataset = ds_cls(device = device, cache_key = cache_key, **data_config)

    if get_rank() == 0:
        logger.warning("%s dataset" % (cls if "version" not in cfg.dataset else f'{cls}({cfg.dataset.version})'))
        if cls != "JointDataset":
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                        (dataset[0].target_edge_index.shape[1], dataset[1].target_edge_index.shape[1],
                            dataset[2].target_edge_index.shape[1]))
        else:
            logger.warning("#train: %d, #valid: %d, #test: %d" %
                           (sum(d.target_edge_index.shape[1] for d in dataset._data[0]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[1]),
                            sum(d.target_edge_index.shape[1] for d in dataset._data[2]),
                            ))

    return dataset



def static_positional_encoding(max_arity, input_dim):
    """
    Generate a static positional encoding.

    Args:
    - max_arity (int): Maximum arity for which to create positional encodings.
    - input_dim (int): Dimension of the input feature vector.

    Returns:
    - torch.Tensor: A tensor containing positional encodings for each position.
    """
    # Initialize the positional encoding matrix
    position = torch.zeros(max_arity + 1, input_dim)

    # Compute the positional encodings
    for pos in range(max_arity + 1):
        # position[pos, pos] = 1
        for i in range(0, input_dim, 2):
            position[pos, i] = np.sin(pos / (10000 ** ((2 * i) / input_dim)))
            if i + 1 < input_dim:
                position[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / input_dim)))


    return position

def coo_to_csr(row, col, edge_types, num_nodes=None):
    if num_nodes is None:
        num_nodes = int(row.max()) + 1

    row, perm = index_sort(row, max_value=num_nodes)
    col = col[perm]
    types = edge_types[perm]

    rowptr = index2ptr(row, num_nodes)
    return rowptr, col, types

def coo_to_csr_hyper(row, col, edge_types, pos_index, num_nodes=None):
    # The only difference is that now col is a 2D tensor
    # Row is the source node, col is the destination node list. 
    if num_nodes is None:
        num_nodes = int(row.max()) + 1

    row, perm = index_sort(row, max_value=num_nodes)
    col = col[:,perm] # 
    types = edge_types[perm]
    pos_index = pos_index[:, perm]
    rowptr = index2ptr(row, num_nodes)
    return rowptr, col, types, pos_index


def smart_split(edge_index):
    max_arity = edge_index.shape[0]
    file = torch.cat([
            torch.cat([edge_index[:arity,:], edge_index[arity+1:,:]], dim = 0) # exclude the current arity
            for arity in range(max_arity)]
            , dim = 1
            )
    return file

def preprocess_triton_hypergraph(edge_index, edge_type, num_node):
    max_arity = edge_index.shape[0]
    destination = edge_index.flatten()
    source = smart_split(edge_index)
    edge_type = edge_type.repeat(max_arity) # expand as if destination

    # Apply the sequence tensor to the non-zero elements
    pos_node_in_edge = torch.arange(1, max_arity+1).unsqueeze(1).repeat(1, edge_index.shape[1]).to(edge_index.device)
    pos_index = smart_split(pos_node_in_edge)

    assert pos_index.shape == source.shape, "pos_index and source should have the same shape"
    # Remove the destination node that is 0
    mask = destination != 0
    destination = destination[mask]
    source = source[:, mask]
    edge_type = edge_type[mask]
    pos_index  = pos_index[:, mask]

    rowptr, indices, etypes, pos_index = coo_to_csr_hyper(destination, source, edge_type, pos_index, num_node)

    num_rel = edge_type.max().item() + 1
    # also create a tensor of shape [num_node, relation] to indicate the source node and incoming degree of edge type

    node_edge_type_degree = torch.sparse_coo_tensor(
        torch.stack([destination, edge_type], dim = 0),
        torch.ones_like(edge_type), device = edge_index.device).coalesce().to_dense().transpose(0,1)
    assert node_edge_type_degree.stride() == (1, num_rel), f"node_edge_type_degree should have stride (1, num_rels), but have {node_edge_type_degree.stride()} instead"

    return rowptr, indices, etypes, pos_index, node_edge_type_degree
