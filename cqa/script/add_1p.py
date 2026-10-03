#!/usr/bin/env python3
"""Copy the 1p (base link-prediction) query type from the shipped/original datasets into every
+H benchmark, for BOTH the test and valid splits. 1p has no reduction tiers (a single held-out
hop IS full-inference), so it is included as-is — NOT complexity-balanced and NOT subject to the
frequency cap (it is the reference single-hop task). We (a) merge 1p into the flat {split}_queries.pkl +
{split}_answers_{easy,hard}.pkl, (b) add a {split}-query-reduction/1p/1p/ folder so --test
reductions evaluates it too, and (c) append a 1p row to the {split} reduction_stats.{csv,json}.

Idempotent: re-running overwrites the 1p entries/folder with the same shipped data.

Usage: python cqa/script/add_1p.py --dataroot DR
DR holds each original benchmark next to its +H version (e.g. DR/550 and DR/550H,
DR/WikiTopics_QE/tax and DR/WikiTopics_QE/taxH).
"""
import os, csv, json, glob, pickle, argparse

ONE_P = ('e', ('r',))


def load(p):
    return pickle.load(open(p, "rb")) if os.path.exists(p) else None


def dump(o, p):
    with open(p, "wb") as f:
        pickle.dump(o, f, protocol=pickle.HIGHEST_PROTOCOL)


def add_1p_split(hdir, base, split):
    sq = load(os.path.join(base, f"{split}_queries.pkl"))
    se = load(os.path.join(base, f"{split}_answers_easy.pkl"))
    sh = load(os.path.join(base, f"{split}_answers_hard.pkl"))
    if sq is None or ONE_P not in sq or se is None or sh is None:
        return None
    q1 = set(sq[ONE_P])
    e1 = {q: set(se[ONE_P].get(q, set())) for q in q1}
    h1 = {q: set(sh[ONE_P].get(q, set())) for q in q1}

    # (a) merge into the flat +H files (loader reads all splits)
    for fn, entry in [(f"{split}_queries.pkl", q1),
                      (f"{split}_answers_easy.pkl", e1),
                      (f"{split}_answers_hard.pkl", h1)]:
        p = os.path.join(hdir, fn)
        d = load(p)
        if d is None:
            return None
        d[ONE_P] = entry
        dump(d, p)

    # (b) stratified reduction folder  {split}-query-reduction/1p/1p/
    folder = os.path.join(hdir, f"{split}-query-reduction", "1p", "1p")
    os.makedirs(folder, exist_ok=True)
    dump({ONE_P: q1}, os.path.join(folder, f"{split}_queries.pkl"))
    dump({ONE_P: e1}, os.path.join(folder, f"{split}_answers_easy.pkl"))
    dump({ONE_P: h1}, os.path.join(folder, f"{split}_answers_hard.pkl"))

    # (c) stats
    n_pairs = sum(len(v) for v in h1.values())
    row = {"type": "1p", "tier": "1p", "queries": len(q1), "qa_pairs": n_pairs}
    base_name = "reduction_stats" if split == "test" else f"{split}_reduction_stats"
    cpath = os.path.join(hdir, base_name + ".csv")
    jpath = os.path.join(hdir, base_name + ".json")
    rows = []
    if os.path.exists(cpath):
        rows = [r for r in csv.DictReader(open(cpath)) if not (r["type"] == "1p")]
    rows = [{"type": r["type"], "tier": r["tier"], "queries": int(r["queries"]),
             "qa_pairs": int(r["qa_pairs"])} for r in rows] + [row]
    rows.sort(key=lambda r: (r["type"], r["tier"]))
    with open(cpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "tier", "queries", "qa_pairs"])
        w.writeheader(); w.writerows(rows)
    with open(jpath, "w") as f:
        json.dump(rows, f, indent=2)
    return len(q1), n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    args = ap.parse_args()
    DR = args.dataroot
    benches = []
    for d in sorted(glob.glob(os.path.join(DR, "*H"))):
        v = os.path.basename(d)[:-1]
        if v.isdigit():
            benches.append((d, os.path.join(DR, v)))
    for d in sorted(glob.glob(os.path.join(DR, "WikiTopics_QE", "*H"))):
        g = os.path.basename(d)[:-1]
        benches.append((d, os.path.join(DR, "WikiTopics_QE", g)))

    print(f"Adding 1p to {len(benches)} benchmarks (test + valid)\n")
    for hdir, base in benches:
        name = os.path.relpath(hdir, DR)
        out = []
        for split in ["test", "valid"]:
            r = add_1p_split(hdir, base, split)
            out.append(f"{split}={r[0]}q/{r[1]}p" if r else f"{split}=SKIP")
        print(f"  {name:24s} 1p: " + "  ".join(out))


if __name__ == "__main__":
    main()
