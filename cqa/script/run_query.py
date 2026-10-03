import os
import sys
import csv
import math
import time
import pickle
import pprint
import argparse
from itertools import islice
from tqdm import tqdm

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import datasets_query, tasks, util, query_utils
from ultra.models import Ultra
from ultra.ultraquery import UltraQuery
from ultra.query_utils import batch_evaluate, evaluate, gather_results, Query
from ultra.base_nbfnet import index_to_mask
from ultra.variadic import variadic_softmax
from timeit import default_timer as timer


separator = ">" * 30
line = "-" * 30


def build_query_model(cfg):
    """Build MeritQuery or UltraQuery from cfg and initialize it from the configured
    checkpoints. The model is returned on CPU; the caller moves it to the device."""
    if cfg.model["class"] == "MeritQuery":
        # imported here: MERIT loads kgfm's kernels, whose extension shares the name
        # "rspmm" with the one UltraQuery uses
        from ultra.meritquery import MeritQuery, MeritBackbone
        backbone = MeritBackbone(rel_model_cfg=cfg.model.model.relation_model,
                                 entity_model_cfg=cfg.model.model.entity_model)
        query_cls = MeritQuery
    elif cfg.model["class"] == "UltraQuery":
        backbone = Ultra(rel_model_cfg=cfg.model.model.relation_model,
                         entity_model_cfg=cfg.model.model.entity_model)
        query_cls = UltraQuery
    else:
        raise ValueError(f"unknown model class {cfg.model['class']!r}")
    model = query_cls(
        model=backbone,
        logic=cfg.model.logic,
        dropout_ratio=cfg.model.dropout_ratio,
        threshold=cfg.model.threshold,
        more_dropout=cfg.model.get('more_dropout', 0.0),
    )

    # initialize the backbone from a pretrained link-prediction checkpoint (MERIT or ULTRA)
    if cfg.get("backbone_ckpt") is not None:
        state = torch.load(cfg.backbone_ckpt, map_location="cpu")
        model.model.model.load_state_dict(state["model"])

    # initialize from a fine-tuned query-answering checkpoint
    if cfg.get("query_ckpt") is not None:
        state = torch.load(cfg.query_ckpt, map_location="cpu")
        model.load_state_dict(state["model"])

    return model


def predict_and_target(model, graph, batch):
    query = batch["query"]
    type = batch["type"]
    easy_answer = batch["easy_answer"]
    hard_answer = batch["hard_answer"]
    
    # turn off symbolic traversal at inference time
    pred = model(graph, query, symbolic_traversal=model.training)
    if not model.training:
        # eval
        target = (type, easy_answer, hard_answer)
        restrict_nodes = getattr(graph, "restrict_nodes", None)
        ranking, answer_ranking = batch_evaluate(pred, target, restrict_nodes)
        # answer set cardinality prediction
        prob = F.sigmoid(pred)
        num_pred = (prob * (prob > 0.5)).sum(dim=-1)
        num_easy = easy_answer.sum(dim=-1)
        num_hard = hard_answer.sum(dim=-1)
        return (ranking, num_pred), (type, answer_ranking, num_easy, num_hard)
    else:
        target = easy_answer.float()

    return pred, target

def train_and_validate(cfg, model, train_graph, train_data, valid_graph, valid_data, query_id2type, device, logger, batch_per_epoch=None):
    if cfg.train.num_epoch == 0:
        return

    world_size = util.get_world_size()
    rank = util.get_rank()

    sampler = torch_data.DistributedSampler(train_data, world_size, rank)
    train_loader = torch_data.DataLoader(train_data, cfg.train.batch_size, sampler=sampler)

    batch_per_epoch = batch_per_epoch or len(train_loader)

    cls = cfg.optimizer.pop("class")
    optimizer = getattr(optim, cls)(model.parameters(), **cfg.optimizer)
    num_params = sum(p.numel() for p in model.parameters())
    logger.warning(line)
    logger.warning(f"Number of parameters: {num_params}")

    if world_size > 1:
        parallel_model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
    else:
        parallel_model = model

    step = math.ceil(cfg.train.num_epoch / 10)
    last_epoch = -1

    batch_id = 0
    for i in range(0, cfg.train.num_epoch, step):
        parallel_model.train()
        for epoch in range(i, min(cfg.train.num_epoch, i + step)):
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Epoch %d begin" % epoch)

            losses = []
            sampler.set_epoch(epoch)
            for batch in islice(train_loader, batch_per_epoch):
                if device.type == "cuda":
                    train_graph = train_graph.to(device)
                    batch = query_utils.cuda(batch, device=device)
                pred, target = predict_and_target(parallel_model, train_graph, batch)

                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")

                is_positive = target > 0.5
                is_negative = target <= 0.5
                num_positive = is_positive.sum(dim=-1)
                num_negative = is_negative.sum(dim=-1)

                neg_weight = torch.zeros_like(pred)
                neg_weight[is_positive] = (1 / num_positive.float()).repeat_interleave(num_positive)

                if cfg.task.adversarial_temperature > 0:
                    with torch.no_grad():
                        logit = pred[is_negative] / cfg.task.adversarial_temperature
                        neg_weight[is_negative] = variadic_softmax(logit, num_negative)
                else:
                    neg_weight[is_negative] = (1 / num_negative.float()).repeat_interleave(num_negative)
                loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
                loss = loss.mean()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if util.get_rank() == 0 and batch_id % cfg.train.log_interval == 0:
                    logger.warning(separator)
                    logger.warning("binary cross entropy: %g" % loss)
                losses.append(loss.item())
                batch_id += 1

            if util.get_rank() == 0:
                avg_loss = sum(losses) / len(losses)
                logger.warning(separator)
                logger.warning("Epoch %d end" % epoch)
                logger.warning(line)
                logger.warning("average binary cross entropy: %g" % avg_loss)

        epoch = min(cfg.train.num_epoch, i + step)
        if rank == 0:
            logger.warning("Save checkpoint to model_epoch_%d.pth" % epoch)
            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            torch.save(state, "model_epoch_%d.pth" % epoch)
        util.synchronize()

        if rank == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        test(cfg, model, valid_graph, valid_data, query_id2type=query_id2type, device=device, logger=logger)
        # CQA fine-tuning keeps the last epoch
        last_epoch = epoch

    if rank == 0:
        logger.warning("Load checkpoint from model_epoch_%d.pth" % last_epoch)
    state = torch.load("model_epoch_%d.pth" % last_epoch, map_location=device)
    model.load_state_dict(state["model"])
    util.synchronize()


@torch.no_grad()
def test(cfg, model, test_graph, test_data, query_id2type, device, logger, return_metrics=False):
    world_size = util.get_world_size()
    rank = util.get_rank()

    # Deterministic eval: MERIT's relation encoder draws random node features (RNI) on
    # every forward, so reseed at the start of every test() for reproducible numbers.
    eval_seed = int(cfg.get("eval_seed", 1024))
    torch.manual_seed(eval_seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(eval_seed + rank)

    sampler = torch_data.DistributedSampler(test_data, world_size, rank)
    test_loader = torch_data.DataLoader(test_data, cfg.train.batch_size, sampler=sampler)

    model.eval()
    preds, targets = [], []
    for batch in tqdm(test_loader):
        if device.type == "cuda":
            test_graph = test_graph.to(device)
            batch = query_utils.cuda(batch, device=device)
        
        predictions, target = predict_and_target(model, test_graph, batch)
        preds.append(predictions)
        targets.append(target)
    
    pred = query_utils.cat(preds)
    target = query_utils.cat(targets)

    pred, target = gather_results(pred, target, rank, world_size, device)
    
    metrics = {}
    if rank == 0:
        metrics = evaluate(pred, target, cfg.task.metric, query_id2type)
        query_utils.print_metrics(metrics, logger)
    else:
        metrics['mrr'] = (1 / pred[0].float()).mean().item()
    util.synchronize()
    return metrics


def _pload(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class ReductionFolderDataset(torch_data.Dataset):
    """In-memory test set for ONE `test-query-reduction/{type}/{tier}/` folder.

    Reuses the base dataset's already-built `test_graph`, vocab, and padding so the
    per-tier set is scored against the exact same conditioning graph as `--test all`.
    Yields the same item dict as the dataset's __getitem__, so `test()` runs unchanged.
    Handles both layouts: inductive nested (`{split}_answers_{easy,hard}.pkl`,
    {struct:{query:set}}) and transductive flat (`{split}-{easy,hard}-answers.pkl`).
    """

    def __init__(self, base, folder, split="test"):
        self.num_entity = base.test_graph.num_nodes
        self.max_query_length = base.max_query_length
        if os.path.exists(os.path.join(folder, f"{split}_queries.pkl")):      # inductive (nested)
            q = _pload(os.path.join(folder, f"{split}_queries.pkl"))
            ea = _pload(os.path.join(folder, f"{split}_answers_easy.pkl"))
            ha = _pload(os.path.join(folder, f"{split}_answers_hard.pkl"))
            geta = lambda d, st, qq: d[st][qq]
        else:                                                                  # transductive (flat)
            q = _pload(os.path.join(folder, f"{split}-queries.pkl"))
            ea = _pload(os.path.join(folder, f"{split}-easy-answers.pkl"))
            ha = _pload(os.path.join(folder, f"{split}-hard-answers.pkl"))
            geta = lambda d, st, qq: d[qq]
        self.items = []
        for st, qs in q.items():
            tid = base.type2id[base.struct2type[st]]
            for qq in sorted(qs):
                self.items.append((Query.from_nested(qq), tid,
                                   list(geta(ea, st, qq)), list(geta(ha, st, qq))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        query, tid, easy, hard = self.items[i]
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": tid,
            "easy_answer": index_to_mask(torch.tensor(easy, dtype=torch.long), self.num_entity),
            "hard_answer": index_to_mask(torch.tensor(hard, dtype=torch.long), self.num_entity),
        }


def run_reductions(cfg, model, dataset, test_graph, device, logger, results_file):
    """--test reductions: score every `test-query-reduction/{type}/{tier}/` folder against
    the once-built test_graph and write a per-(type, tier) stratified CSV."""
    # Most benchmarks ship `test-query-reduction/`; the official is-cqa ICEWS18+H
    # ships the same <type>/<tier>/test-*.pkl layout under `test-query-reduction-prop/`,
    # plus an aggregate `all` tier per type.
    red_root = None
    for _cand in ("test-query-reduction", "test-query-reduction-prop"):
        _p = os.path.join(dataset.raw_dir, _cand)
        if os.path.isdir(_p):
            red_root = _p
            break
    if red_root is None:
        raise FileNotFoundError(
            f"No reduction folders at {os.path.join(dataset.raw_dir, 'test-query-reduction[-prop]')}; "
            f"build them with cqa/script/build_reductions.py")
    if util.get_rank() == 0:
        logger.warning("[reductions] using %s" % red_root)
    rows = []
    for type_name in sorted(os.listdir(red_root)):
        tdir = os.path.join(red_root, type_name)
        if not os.path.isdir(tdir):
            continue
        for tier_name in sorted(os.listdir(tdir)):
            folder = os.path.join(tdir, tier_name)
            if not os.path.isdir(folder):
                continue
            fds = ReductionFolderDataset(dataset, folder)
            if len(fds) == 0:
                continue
            if util.get_rank() == 0:
                logger.warning("%s\n[reductions] %s/%s: %d queries"
                               % (separator, type_name, tier_name, len(fds)))
            m = test(cfg, model, test_graph, fds, query_id2type=dataset.id2type,
                     device=device, logger=logger)
            if util.get_rank() == 0:
                row = {"dataset": str(dataset), "type": type_name, "tier": tier_name,
                       "queries": len(fds)}
                for metric_name in cfg.task.metric:
                    row[metric_name] = m.get("[%s] %s" % (type_name, metric_name), float("nan"))
                rows.append(row)
    if util.get_rank() == 0 and rows:
        fields = ["dataset", "type", "tier", "queries"] + list(cfg.task.metric)
        with open(results_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        logger.warning("Wrote stratified (per type/tier) results -> %s" % results_file)
    util.synchronize()
    return rows


if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    # the eval RNI seed is this run's seed (test() reads cfg.eval_seed)
    cfg.eval_seed = args.seed
    # --test selects the eval mode (util's parsers use parse_known_args, so this
    # extra flag passes through them harmlessly and we recover it here).
    _tp = argparse.ArgumentParser()
    _tp.add_argument("--test", default="all", choices=["all", "reductions", "both"],
                     help="all = whole test set in one pass; reductions = loop the "
                          "per-(type,tier) test-query-reduction folders (stratified); "
                          "both = all then reductions in ONE process (same numbers, one setup)")
    test_mode = _tp.parse_known_args()[0].test
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + util.get_rank())

    logger = util.get_root_logger()
    if util.get_rank() == 0:
        logger.warning("Random seed: %d" % args.seed)
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))
    
    task_name = cfg.task["name"]
    dataset = query_utils.build_query_dataset(cfg)
    device = util.get_device(cfg)
    results_file = os.path.join(cfg.output_dir, f"results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
    
    train_data, valid_data, test_data = dataset.split()
    train_graph, valid_graph, test_graph = dataset.train_graph, dataset.valid_graph, dataset.test_graph

    model = build_query_model(cfg)

    if "fast_test" in cfg.train:
        if util.get_rank() == 0:
            logger.warning("Quick test mode on. Only evaluate on %d samples for valid" % cfg.train.fast_test)
        g = torch.Generator()
        g.manual_seed(1024)
        valid_data = torch_data.random_split(valid_data, [cfg.train.fast_test, len(valid_data) - cfg.train.fast_test], generator=g)[0]
        

    model = model.to(device)

    # `both` runs the two eval modes in one process; test() reseeds at every call, so the
    # numbers equal those of two separate runs.
    if test_mode in ("all", "both"):
        # --test all: train (no-op at epochs=0) + valid + whole-test in one pass
        train_and_validate(cfg, model, train_graph, train_data, valid_graph, valid_data, query_id2type=dataset.id2type, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on valid")
        start = timer()
        val_metrics = test(cfg, model, valid_graph, valid_data, query_id2type=dataset.id2type, device=device, logger=logger)
        end = timer()
        logger.warning(f"Valid time: {end - start}")
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on test")
        metrics = test(cfg, model, test_graph, test_data, query_id2type=dataset.id2type, device=device, logger=logger)
        # write to the log file
        if util.get_rank() == 0:
            metrics['dataset'] = str(dataset)
            query_utils.print_metrics_to_file(metrics, results_file)

    if test_mode in ("reductions", "both"):
        # --test reductions: stratified per-(type,tier) eval over the reduction folders
        if util.get_rank() == 0:
            logger.warning(separator)
            logger.warning("Evaluate on TEST reduction folders (per type/tier, stratified)")
        red_file = os.path.join(cfg.output_dir, f"reductions_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
        run_reductions(cfg, model, dataset, test_graph, device, logger, red_file)
