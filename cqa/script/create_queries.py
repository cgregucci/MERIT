#!/usr/bin/env python3
"""Standalone, torch-free CQA benchmark generator for ULTRA.

Implements the reduction-tier strategy of the +H benchmarks for transductive and
inductive datasets, emitting queries/answers in
ULTRA's on-disk format. The graph that defines hardness is, per setting:
  * transductive: is-cqa cumulative (observed=train / train+valid; held-out=valid/test)
  * inductive: exactly ULTRA's {split}_graph (train_graph + {split}_inference);
    WikiTopics excludes train (disjoint vocab). held-out = {split}_predict.

Per-reduction-group (q,a)-pair quota: N_group = min(10000, round(alpha * T_heldout)),
T_heldout = #held-out triples for the split (alpha=0.5 by default).

Usage:
  python cqa/script/create_queries.py --dataset InductiveFB15k237Query --version 550 --root <data_root> \
      --out_root <out_root> --splits test,valid --balanced --seed 0 --freq_cap_pct 25
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # script/

from cqa_gen import registry, graph_io, sampler, emit, balanced


def count_lines(paths):
    n = 0
    for p in paths:
        with open(p) as f:
            for _ in f:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Standalone CQA benchmark generator for ULTRA")
    ap.add_argument("--dataset", required=True, choices=list(registry.DATASETS),
                    help="dataset key in cqa_gen.registry.DATASETS")
    ap.add_argument("--version", default=None, help="inductive version (e.g. 550 / art)")
    ap.add_argument("--root", required=True, help="root holding the dataset dir")
    ap.add_argument("--out_root", default=None, help="output root (default: --root)")
    ap.add_argument("--out_suffix", default="H", help="suffix for the output dataset dir")
    ap.add_argument("--splits", default="test,valid", help="comma list: test,valid[,train]")
    ap.add_argument("--types", default=None, help="comma list of query types (default: all EPFO+neg)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.5, help="quota = min(10000, round(alpha*T_heldout))")
    ap.add_argument("--gen_num", type=int, default=None, help="override per-group quota directly")
    ap.add_argument("--max_ans_num", type=float, default=1e6)
    ap.add_argument("--max_outer_tries_factor", type=int, default=200,
                    help="cap outer sampling tries at factor*gen_num (0=uncapped)")
    ap.add_argument("--balanced", action="store_true",
                    help="balanced +H: every reduction group = exactly N pairs "
                         "(full tier via held-out grounding, partials via full grounding)")
    ap.add_argument("--max_tries_factor", type=int, default=500,
                    help="[--balanced] cap tries per tier at factor*N")
    ap.add_argument("--freq_cap_pct", type=float, default=25.0,
                    help="[--balanced] max %% of (q,a) pairs any single relation OR anchor may hold "
                         "in a tier (the released +H benchmarks use 25)")
    args = ap.parse_args()

    cfg = registry.DATASETS[args.dataset]
    out_root = args.out_root or args.root
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    types = ([t.strip() for t in args.types.split(",")] if args.types
             else registry.EPFO_TYPES[1:] + registry.NEGATION_TYPES)  # skip 1p

    dirname = cfg.dirname.format(version=args.version) if "{version}" in cfg.dirname else cfg.dirname
    base_dir = os.path.join(args.root, dirname)
    out_dir = os.path.join(out_root, dirname + args.out_suffix)
    os.makedirs(out_dir, exist_ok=True)

    specs = registry.build_graph_specs(args.dataset, splits)
    is_transductive = cfg.setting == "transductive"

    # copy base KG files the loader needs (triples + id maps)
    if is_transductive:
        emit.copy_base_files(base_dir, out_dir,
                             [cfg.train_file, cfg.valid_file, cfg.test_file,
                              "id2ent.pkl", "id2rel.pkl", "stats.txt"])
    else:
        emit.copy_base_files(base_dir, out_dir,
                             [cfg.train_graph_file, cfg.val_inf_file, cfg.test_inf_file])

    for split in splits:
        spec = specs[split]
        views = graph_io.load_graph_views(spec, base_dir)
        t_held = count_lines([os.path.join(base_dir, f) for f in spec.heldout_files])
        if args.balanced:
            # per-query-type N_type = min(target, n_possible under the frequency cap);
            # within a type every reduction group is balanced to N_type. Flat target (10k).
            target = args.gen_num if args.gen_num else 10000
            print(f"[{args.dataset}{('/'+args.version) if args.version else ''}] split={split} "
                  f"T_heldout={t_held} -> target={target} (BALANCED per-type; N_type=min(target,n_possible))")
            q, filt, hard, tier_data = balanced.generate_balanced(
                types, views, target=target, seed=args.seed, max_ans_num=args.max_ans_num,
                freq_cap_pct=args.freq_cap_pct, max_tries_factor=args.max_tries_factor, log_every=1)
        else:
            gen_num = args.gen_num if args.gen_num else min(10000, round(args.alpha * t_held))
            cap = args.max_outer_tries_factor * gen_num if args.max_outer_tries_factor else 0
            print(f"[{args.dataset}{('/'+args.version) if args.version else ''}] split={split} "
                  f"T_heldout={t_held} -> N_group={gen_num} (alpha={args.alpha})")
            q, filt, hard, tier_data = sampler.ground_queries(
                types, views, gen_num, max_ans_num=args.max_ans_num, seed=args.seed,
                max_outer_tries=cap, log_every=1, return_tiers=True)

        mode = "transductive" if is_transductive else "inductive"
        if is_transductive:
            emit.emit_transductive(out_dir, split, q, filt, hard)
        else:
            emit.emit_inductive(out_dir, split, q, filt, hard)
        tot = sum(len(v) for v in hard.values())
        print(f"   wrote split={split}: {sum(len(s) for s in q.values())} queries, "
              f"{tot} (q,a) pairs -> {out_dir}")

        # per-(type, reduction-tier) stratified folders + stats. Written per generated split:
        # `{split}-query-reduction/` + `{split==test?reduction_stats:valid_reduction_stats}.{csv,json}`.
        if split in ("test", "valid"):
            emit.emit_reduction_folders(out_dir, split, tier_data, hard, filt,
                                        registry.TYPE2STRUCT, mode)
            basename = "reduction_stats" if split == "test" else f"{split}_reduction_stats"
            stats = emit.dump_reduction_stats(out_dir, tier_data, basename=basename)
            print(f"   [{split}] reduction tiers (queries/pairs): " + ", ".join(
                f"{r['type']}/{r['tier']}={r['queries']}q/{r['qa_pairs']}p" for r in stats))

    print("DONE ->", out_dir)


if __name__ == "__main__":
    main()
