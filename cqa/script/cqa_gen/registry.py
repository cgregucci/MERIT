"""Torch-free constants + per-dataset registry for the CQA generator.

`STRUCT2TYPE` mirrors `cqa/ultra/datasets_query.py` exactly (a plain tuple/str
constant). The `DATASETS` registry + `build_graph_specs` encode
the per-split "which graph defines hardness" choice.
"""
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Query-type taxonomy (exact copy of cqa/ultra/datasets_query.py)
# ---------------------------------------------------------------------------
STRUCT2TYPE = {
    ("e", ("r",)): "1p",
    ("e", ("r", "r")): "2p",
    ("e", ("r", "r", "r")): "3p",
    ("e", ("r", "r", "r", "r")): "4p",                                  # new (is-cqa +H)
    (("e", ("r",)), ("e", ("r",))): "2i",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "4i",  # new (is-cqa +H)
    ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
    (("e", ("r", "r")), ("e", ("r",))): "pi",
    (("e", ("r",)), ("e", ("r", "n"))): "2in",
    (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
    ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
    (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
    (("e", ("r", "r", "n")), ("e", ("r",))): "pni",
    (("e", ("r",)), ("e", ("r",)), ("u",)): "2u-DNF",
    ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up-DNF",
    ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",)): "2u-DM",
    ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r")): "up-DM",
}
TYPE2STRUCT = {v: k for k, v in STRUCT2TYPE.items()}

# Number of *partial* reduction tiers per type (full-inference is the implicit +1
# group). ULTRA emits DNF unions only.
GEN_NUM_PER_QUERY = {
    "1p": [], "2p": [0], "3p": [0, 0], "4p": [0, 0, 0],
    "2i": [0], "3i": [0, 0], "4i": [0, 0, 0],
    "pi": [0, 0, 0], "ip": [0, 0, 0],
    "2u-DNF": [], "up-DNF": [0, 0],
    "2in": [], "3in": [0], "pin": [0], "pni": [], "inp": [0],
}

# Nested-list structure per type (the empty template fed to fill_query / the
# classifier). 'e'=anchor, 'r'=relation, 'n'=negation, 'u'=union marker.
TYPE2STRUCT_LIST = {
    "1p": ["e", ["r"]],
    "2p": ["e", ["r", "r"]],
    "3p": ["e", ["r", "r", "r"]],
    "4p": ["e", ["r", "r", "r", "r"]],
    "2i": [["e", ["r"]], ["e", ["r"]]],
    "3i": [["e", ["r"]], ["e", ["r"]], ["e", ["r"]]],
    "4i": [["e", ["r"]], ["e", ["r"]], ["e", ["r"]], ["e", ["r"]]],
    "pi": [["e", ["r", "r"]], ["e", ["r"]]],
    "ip": [[["e", ["r"]], ["e", ["r"]]], ["r"]],
    "2u-DNF": [["e", ["r"]], ["e", ["r"]], ["u"]],
    "up-DNF": [[["e", ["r"]], ["e", ["r"]], ["u"]], ["r"]],
    "2in": [["e", ["r"]], ["e", ["r", "n"]]],
    "3in": [["e", ["r"]], ["e", ["r"]], ["e", ["r", "n"]]],
    "pin": [["e", ["r", "r"]], ["e", ["r", "n"]]],
    "pni": [["e", ["r", "r", "n"]], ["e", ["r"]]],
    "inp": [[["e", ["r"]], ["e", ["r", "n"]]], ["r"]],
}

EPFO_TYPES = ["1p", "2p", "3p", "4p", "2i", "3i", "4i", "pi", "ip", "2u-DNF", "up-DNF"]
NEGATION_TYPES = ["2in", "3in", "pin", "pni", "inp"]

# Markers used in grounded queries / achieve_answer.
NEGATION = -2
UNION = -1


# ---------------------------------------------------------------------------
# GraphSpec: the per-split "which graph defines hardness" abstraction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GraphSpec:
    """Files (relative to the dataset dir) whose union forms each graph view.

    observed = the conditioning graph the model sees (transductive: is-cqa
    cumulative; inductive: ULTRA's `{split}_graph`). heldout = the to-predict
    edges. full = observed ∪ heldout (derived). universe_mode selects the
    negation/answer candidate set: 'all' (transductive, all entities) or
    'observed' (inductive, restrict_nodes = nodes of observed).
    """
    split: str                       # "train" | "valid" | "test"
    observed_files: tuple            # files unioned -> observed adjacency
    heldout_files: tuple             # files unioned -> held-out adjacency
    inverse: str                     # "plus_one" (BetaE) | "plus_half" (inductive)
    universe_mode: str               # "all" | "observed"


@dataclass(frozen=True)
class DatasetCfg:
    name: str
    setting: str                     # "transductive" | "inductive" | "inductive_disjoint"
    dirname: str                     # may contain "{version}"
    inverse: str                     # "plus_one" | "plus_half"
    # transductive triple files (already symmetrized, int h\tr\tt):
    train_file: str = "train.txt"
    valid_file: str = "valid.txt"
    test_file: str = "test.txt"
    has_id_maps: bool = True
    # inductive triple/predict files:
    train_graph_file: str = "train_graph.txt"
    val_inf_file: str = "val_inference.txt"
    test_inf_file: str = "test_inference.txt"
    val_predict_file: Optional[str] = "val_predict.txt"
    test_predict_file: Optional[str] = "test_prediction.txt"   # WikiTopics name; FB overrides
    include_train_in_observed: bool = True   # False for WikiTopics (disjoint vocab)


DATASETS = {
    # --- transductive (is-cqa cumulative strategy, as for FB15k237+H) ---
    "FB15k-237-betae": DatasetCfg(name="FB15k-237-betae", setting="transductive",
                                  dirname="FB15k-237-betae", inverse="plus_one"),
    "NELL-betae": DatasetCfg(name="NELL-betae", setting="transductive",
                             dirname="NELL-betae", inverse="plus_one"),
    "FB15k-betae": DatasetCfg(name="FB15k-betae", setting="transductive",
                              dirname="FB15k-betae", inverse="plus_one"),
    # --- inductive (entity): match ULTRA conditioning graph 1:1 ---
    "InductiveFB15k237Query": DatasetCfg(
        name="InductiveFB15k237Query", setting="inductive", dirname="{version}",
        inverse="plus_half", has_id_maps=False,
        val_predict_file="val_predict.txt", test_predict_file="test_predict.txt",
        include_train_in_observed=True),
    # --- inductive (entity+relation): disjoint test graph, exclude train ---
    "WikiTopicsQuery": DatasetCfg(
        name="WikiTopicsQuery", setting="inductive_disjoint",
        dirname="WikiTopics_QE/{version}", inverse="plus_half", has_id_maps=False,
        val_predict_file="val_prediction.txt", test_predict_file="test_prediction.txt",
        include_train_in_observed=False),
}


def build_graph_specs(dataset: str, splits) -> dict:
    """Return {split: GraphSpec} encoding observed/heldout/universe per split.

    Transductive (is-cqa cumulative): valid observed=train, heldout=valid;
    test observed=train+valid, heldout=test. Inductive: observed = ULTRA's
    `{split}_graph` (train_graph + {split}_inference, or test_inference alone
    for the disjoint WikiTopics), heldout = {split}_predict.
    """
    cfg = DATASETS[dataset]
    specs = {}
    for split in splits:
        if cfg.setting == "transductive":
            if split == "valid":
                obs, held = (cfg.train_file,), (cfg.valid_file,)
            elif split == "test":
                obs, held = (cfg.train_file, cfg.valid_file), (cfg.test_file,)
            elif split == "train":
                # train queries: observed = train, no held-out (handled separately)
                obs, held = (cfg.train_file,), ()
            else:
                raise ValueError(split)
            specs[split] = GraphSpec(split, obs, held, cfg.inverse, "all")
        else:  # inductive / inductive_disjoint
            inf = {"valid": cfg.val_inf_file, "test": cfg.test_inf_file,
                   "train": cfg.train_graph_file}[split]
            pred = {"valid": cfg.val_predict_file, "test": cfg.test_predict_file,
                    "train": None}[split]
            if split == "train":
                obs, held = (cfg.train_graph_file,), ()
            elif cfg.include_train_in_observed:
                obs, held = (cfg.train_graph_file, inf), ((pred,) if pred else ())
            else:  # WikiTopics: train excluded (disjoint vocab)
                obs, held = (inf,), ((pred,) if pred else ())
            specs[split] = GraphSpec(split, obs, held, cfg.inverse, "observed")
    return specs
