"""Balanced +H generator with a hard relation/anchor frequency cap (--freq_cap_pct).

Per query type, every reduction group holds exactly N_type = min(target, per-tier max achievable
under the cap) pairs; N_type differs across types. We do NOT over-collect and trim.
Instead, for each tier we find the MAXIMUM number of (q,a) pairs that can be packed while keeping
every canonical relation and every anchor within the cap — a greedy build-up (each
query contributes up to the min headroom over all its relation/anchor keys), maximized by binary
search on N. N_type = min tier max; then every tier is packed to exactly N_type the same way, so
the result is balanced within-type and cap-clean. Small/relation-poor graphs therefore
yield small (but maximal, balanced, cap-clean) groups.

Returns (queries, filtered_answers, hard_answers, tier_data) in sampler.ground_queries' shape.
"""
import time
import random
import logging
from collections import defaultdict

from . import reduction
from . import enumerate_tiers
from .registry import TYPE2STRUCT_LIST
from .sampler import achieve_answer, list2tuple, query_rels_anchs


def _prep(group, canon_rel):
    """[(query, answers_list, keys)] where keys = frozenset of ('r',canon_rel) & ('a',anchor)."""
    items = []
    for q, a in group.items():
        if not a:
            continue
        rels, anchs = query_rels_anchs(q)
        keys = frozenset([("r", canon_rel(r)) for r in rels] + [("a", e) for e in anchs])
        items.append((q, list(a), keys))
    return items


def _greedy_fill(items, N, cap_count):
    """Pack up to N pairs so every key is used <= cap_count times. Each query contributes up to
    min(headroom over its keys, remaining, #answers). Returns ({query:set(answers)}, total)."""
    used = defaultdict(int)
    total = 0
    out = {}
    for q, ans, keys in items:
        if total >= N:
            break
        head = N - total
        for k in keys:
            h = cap_count - used[k]
            if h < head:
                head = h
            if head <= 0:
                break
        if head <= 0:
            continue
        take = head if head < len(ans) else len(ans)
        if take <= 0:
            continue
        out[q] = set(ans[:take])
        for k in keys:
            used[k] += take
        total += take
    return out, total


def _max_cap_fill(items, cap_frac, hi):
    """Largest N in [5, hi] for which a cap-clean packing of exactly N pairs exists (cap_count =
    floor(cap_frac*N), so every key <= cap_frac*N). Binary search. Returns (selection, N)."""
    if hi < 5:
        return {}, 0
    lo, high = 5, hi
    best = ({}, 0)
    while lo <= high:
        mid = (lo + high) // 2
        cap = int(cap_frac * mid)
        out, total = _greedy_fill(items, mid, cap) if cap >= 1 else ({}, -1)
        if total >= mid:                      # feasible at mid
            best = (out, mid)
            lo = mid + 1
        else:
            high = mid - 1
    return best


def generate_balanced(types, views, target=10000, seed=0, max_ans_num=1e6, freq_cap_pct=20.0,
                      max_tries_factor=0, log_every=0):
    """Per-type balanced +H under a hard frequency cap, maximizing pairs (no trim). For each type:
    enumerate every tier, find each tier's max cap-clean size, N_type = min of those, then pack
    every tier to exactly N_type."""
    universe = views.universe
    cap_frac = freq_cap_pct / 100.0
    canon_rel = enumerate_tiers._canon_rel(views.inverse, views.num_rel)
    queries = defaultdict(set)
    hard = defaultdict(set)
    filtered = defaultdict(set)
    tier_data = defaultdict(lambda: defaultdict(dict))
    answer_full = {}
    random.seed(seed)

    for type_name in types:
        struct_key = list2tuple(TYPE2STRUCT_LIST[type_name])
        t0 = time.time()
        groups = enumerate_tiers.enumerate_all_tiers(type_name, views, target,
                                                     freq_cap_pct=freq_cap_pct)
        prepped = {tn: _prep(g, canon_rel) for tn, g in groups.items()}
        # per-tier maximum cap-clean size
        maxes = {}
        for tn, items in prepped.items():
            avail = sum(len(a) for _, a, _ in items)
            _, s = _max_cap_fill(items, cap_frac, min(target, avail))
            maxes[tn] = s
        n_type = min([target] + [s for s in maxes.values() if s > 0]) if any(maxes.values()) else 0
        # Pack every tier to exactly n_type. Greedy at a reduced n_type can undershoot, so lower
        # n_type to the worst tier's achieved count and re-pack until all tiers hit it exactly.
        packed = {}
        while n_type > 0:
            packed = {tn: _greedy_fill(items, n_type, int(cap_frac * n_type))
                      for tn, items in prepped.items()}
            got_min = min((got for _, got in packed.values()), default=0)
            if got_min >= n_type:
                break
            n_type = got_min
        for tn, (sel, _got) in packed.items():
            for qt, ans in sel.items():
                if not ans:
                    continue
                tier_data[type_name][tn][qt] = set(ans)
                queries[struct_key].add(qt)
                hard[qt] |= ans
                if qt not in answer_full:
                    answer_full[qt] = achieve_answer(enumerate_tiers._t2l(qt),
                                                     views.full_in, views.full_out, universe)
        logging.info("balanced[%s] N_type=%d  (per-tier max %s)  %.0fs",
                     type_name, n_type, maxes, time.time() - t0)

    for qt in hard:
        filtered[qt] = answer_full.get(qt, set()) - hard[qt]
    return queries, filtered, hard, tier_data
