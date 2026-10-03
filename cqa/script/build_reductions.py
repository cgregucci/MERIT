#!/usr/bin/env python3
"""Rebuild the per-(type, reduction-tier) folders of an inductive +H benchmark.

`run_query.py --test reductions` scores every `{split}-query-reduction/{type}/{tier}/`
folder separately. The tier of a (query, answer) pair is a deterministic function of the
query and the observed / held-out graphs, so the folders are rebuilt from the shipped
valid/test queries instead of being distributed. The held-out triples come with the
original benchmark release (`val_predict.txt` / `test_predict.txt` for InductiveFB15k237,
`val_prediction.txt` / `test_prediction.txt` for WikiTopics).

Usage:
  python cqa/script/build_reductions.py --dataset InductiveFB15k237Query --version 550 \
      --base <dir with the original 550/ files> --plus_h cqa/benchmarks/550H
  python cqa/script/build_reductions.py --dataset WikiTopicsQuery --version tax \
      --base <dir with the original WikiTopics_QE/tax/ files> --plus_h cqa/benchmarks/WikiTopics_QE/taxH
"""
import argparse
import os
import pickle
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # script/

from cqa_gen import registry, graph_io, emit, reduction
from cqa_gen.sampler import achieve_answer


def tuple2list(q):
    return [tuple2list(x) for x in q] if isinstance(q, tuple) else q


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_split(dataset, split, base_dir, plus_h):
    queries = _load(os.path.join(plus_h, f"{split}_queries.pkl"))
    easy = _load(os.path.join(plus_h, f"{split}_answers_easy.pkl"))
    hard = _load(os.path.join(plus_h, f"{split}_answers_hard.pkl"))

    # the graphs: the +H dir carries the observed graph, the base dir the held-out triples
    spec = registry.build_graph_specs(dataset, [split])[split]
    views = graph_io.load_graph_views(spec, base_dir)

    tier_data = defaultdict(lambda: defaultdict(dict))
    flat_hard, flat_easy = {}, {}
    for struct, qs in queries.items():
        type_name = registry.STRUCT2TYPE[struct]
        full_name = reduction.full_tier_name(type_name)
        partial_names = reduction.reported_partial_names(type_name)
        for q in sorted(qs):
            q_hard = set(hard[struct][q])
            flat_hard[q] = q_hard
            flat_easy[q] = set(easy[struct][q])
            if type_name == "1p":
                # a single held-out hop is full inference: one tier with every 1p query
                tier_data["1p"]["1p"][q] = q_hard
                continue
            qlist = tuple2list(q)
            answer_set = achieve_answer(qlist, views.full_in, views.full_out, views.universe)
            train_answer_set = achieve_answer(qlist, views.observed_in, views.observed_out, views.universe)
            _, full_set, partial_sets, _, _ = reduction.compute_fi_pi_answers(
                type_name, q, views, train_answer_set, answer_set)
            if full_set & q_hard:
                tier_data[type_name][full_name][q] = full_set & q_hard
            for name, tier_set in zip(partial_names, partial_sets):
                if tier_set & q_hard:
                    tier_data[type_name][name][q] = tier_set & q_hard

    out_root = os.path.join(plus_h, f"{split}-query-reduction")
    if os.path.isdir(out_root):
        shutil.rmtree(out_root)
    emit.emit_reduction_folders(plus_h, split, tier_data, flat_hard, flat_easy,
                                registry.TYPE2STRUCT, "inductive")
    n = sum(len(q2s) for tiers in tier_data.values() for q2s in tiers.values())
    print(f"[{split}] {n} (query, tier) entries -> {out_root}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, choices=["InductiveFB15k237Query", "WikiTopicsQuery"])
    ap.add_argument("--version", required=True, help="e.g. 550 or tax")
    ap.add_argument("--base", required=True, help="original benchmark dir with the *_predict(ion).txt files")
    ap.add_argument("--plus_h", required=True, help="the +H benchmark dir (written into)")
    ap.add_argument("--splits", default="test,valid")
    args = ap.parse_args()

    cfg = registry.DATASETS[args.dataset]
    # the observed graph files are read from --base too; the +H dir carries identical copies
    for f in (cfg.train_graph_file, cfg.val_inf_file, cfg.test_inf_file):
        a, b = os.path.join(args.base, f), os.path.join(args.plus_h, f)
        if os.path.exists(a) and os.path.exists(b):
            with open(a, "rb") as fa, open(b, "rb") as fb:
                assert fa.read() == fb.read(), f"{f} differs between --base and --plus_h"
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        build_split(args.dataset, split, args.base, args.plus_h)


if __name__ == "__main__":
    main()
