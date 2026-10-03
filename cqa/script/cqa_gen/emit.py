"""Writers that emit generated queries/answers in ULTRA's exact on-disk layout.

Transductive: flat answers, hyphen filenames + id maps.
Inductive: answers nested by struct, underscore filenames, train -> train_answers_hard.pkl.
Query pickle keys are the struct tuples (== struct2type keys, incl. ('u',) marker);
grounded queries carry -1 (union) / -2 (negation) as in the BetaE format.
"""
import os
import csv
import json
import pickle
import shutil


def _dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _nest_by_struct(queries, flat_answers):
    """Convert flat {query: set} -> nested {struct: {query: set}} using `queries`."""
    nested = {}
    for struct, qset in queries.items():
        d = {}
        for q in qset:
            d[q] = flat_answers.get(q, set())
        nested[struct] = d
    return nested


def emit_transductive(out_dir, split, queries, filtered, hard):
    os.makedirs(out_dir, exist_ok=True)
    queries = {k: set(v) for k, v in queries.items()}
    _dump(queries, os.path.join(out_dir, f"{split}-queries.pkl"))
    if split == "train":
        _dump(dict(hard), os.path.join(out_dir, f"{split}-answers.pkl"))
    else:
        _dump(dict(filtered), os.path.join(out_dir, f"{split}-easy-answers.pkl"))
        _dump(dict(hard), os.path.join(out_dir, f"{split}-hard-answers.pkl"))


def emit_inductive(out_dir, split, queries, filtered, hard):
    os.makedirs(out_dir, exist_ok=True)
    queries = {k: set(v) for k, v in queries.items()}
    _dump(queries, os.path.join(out_dir, f"{split}_queries.pkl"))
    if split == "train":
        _dump(_nest_by_struct(queries, hard),
              os.path.join(out_dir, "train_answers_hard.pkl"))
    else:
        _dump(_nest_by_struct(queries, filtered),
              os.path.join(out_dir, f"{split}_answers_easy.pkl"))
        _dump(_nest_by_struct(queries, hard),
              os.path.join(out_dir, f"{split}_answers_hard.pkl"))


def copy_base_files(src_dir, out_dir, files):
    """Copy base KG files (triples, id maps) the loader needs into out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    for f in files:
        s = os.path.join(src_dir, f)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out_dir, f))


def _emit_tier_folder(folder, split, struct, q2hard, q2easy, mode):
    """Write one `{type}/{tier}/` stratified folder in the dataset's on-disk layout.

    transductive: flat answers + hyphen names; inductive: nested-by-struct + underscore.
    The query pkl is `{struct: set(queries)}` in both.
    """
    os.makedirs(folder, exist_ok=True)
    qset = {struct: set(q2hard.keys())}
    if mode == "transductive":
        _dump(qset, os.path.join(folder, f"{split}-queries.pkl"))
        _dump(dict(q2easy), os.path.join(folder, f"{split}-easy-answers.pkl"))
        _dump(dict(q2hard), os.path.join(folder, f"{split}-hard-answers.pkl"))
    else:
        _dump(qset, os.path.join(folder, f"{split}_queries.pkl"))
        _dump({struct: dict(q2easy)}, os.path.join(folder, f"{split}_answers_easy.pkl"))
        _dump({struct: dict(q2hard)}, os.path.join(folder, f"{split}_answers_hard.pkl"))


def emit_reduction_folders(out_dir, split, tier_data, hard_answers, filtered_answers,
                           type2struct, mode):
    """Write per-(type, reduction-tier) stratified folders under
    `{out_dir}/{split}-query-reduction/{type}/{tier}/`.

    Per query, `hard` = that tier's sampled answers only; `easy` = the FULL true
    answer complement of the tier (= `answer_set - tier_hard`, where
    `answer_set = hard_answers[q] | filtered_answers[q]`), so `easy | hard ==
    answer_set` in every folder — matching ULTRA's filtered-ranking semantics
    (it filters against easy ∪ hard) and is-cqa's `filters = answer_set - tier`.
    """
    root = os.path.join(out_dir, f"{split}-query-reduction")
    for type_name, tiers in tier_data.items():
        struct = type2struct[type_name]
        for tier_name, q2set in tiers.items():
            q2hard = {q: set(s) for q, s in q2set.items() if s}
            if not q2hard:
                continue
            q2easy = {}
            for q in q2hard:
                answer_set = set(hard_answers.get(q, set())) | set(filtered_answers.get(q, set()))
                q2easy[q] = answer_set - q2hard[q]
            _emit_tier_folder(os.path.join(root, type_name, tier_name),
                              split, struct, q2hard, q2easy, mode)


def dump_reduction_stats(out_dir, tier_data, basename="reduction_stats"):
    """Write per-(type, tier) #queries and #(q,a) pairs as json + csv."""
    rows = []
    for type_name, tiers in tier_data.items():
        for tier_name, q2set in tiers.items():
            n_pairs = sum(len(s) for s in q2set.values())
            n_queries = sum(1 for s in q2set.values() if s)
            if n_pairs == 0:
                continue
            rows.append({"type": type_name, "tier": tier_name,
                         "queries": n_queries, "qa_pairs": n_pairs})
    rows.sort(key=lambda r: (r["type"], r["tier"]))
    with open(os.path.join(out_dir, f"{basename}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(out_dir, f"{basename}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "tier", "queries", "qa_pairs"])
        w.writeheader()
        w.writerows(rows)
    return rows
