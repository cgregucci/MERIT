"""Deterministic enumeration of the FULL-INFERENCE set for every EPFO type, from the
held-out graph. Used to fill the full tier to N (random grounding under-yields — it
re-hits the same queries on small graphs).

For each type we generate DIVERSE candidate groundings whose atoms are held-out edges,
then hand each to the reduction classifier (compute_fi_pi_answers) to get the
exact full-inference answer set — so "full" matches the classifier by construction. We
collect (query -> full answers) under the relation/anchor frequency cap until N pairs.
Negation full ("pos-only-miss") is abundant via ordinary sampling and is NOT handled here.
"""
import random
import logging
from collections import defaultdict

from . import reduction
from .sampler import (achieve_answer, list2tuple, add_to_freq_dict,
                      top_k_dict_values_sorting, _canon_rel)

UNION = -1


def _t2l(x):
    return [_t2l(e) for e in x] if isinstance(x, tuple) else x


# --------------------------------------------------------------------------- #
# candidate-grounding generators (yield grounded query tuples, atoms held-out) #
# --------------------------------------------------------------------------- #
def _ho_paths(views, k, cap):
    """Up to `cap` (anchor, relseq[, reach]) held-out k-paths (reach = held-out k-reach)."""
    H = views.heldout_out
    out = []
    anchors = list(H.keys())
    random.shuffle(anchors)
    for a in anchors:
        frontier = {(): {a}}
        for _ in range(k):
            nxt = {}
            for rs, nodes in frontier.items():
                rel2set = defaultdict(set)
                for v in nodes:
                    hv = H.get(v)
                    if hv:
                        for r, ts in hv.items():
                            rel2set[r] |= ts
                for r, s in rel2set.items():
                    nxt[rs + (r,)] = s
            frontier = nxt
        for rs, reach in frontier.items():
            out.append((a, rs, reach))
            if len(out) >= cap:
                return out
    return out


def _path_cands(views, k, cap):
    for a, rs, _ in _ho_paths(views, k, cap):
        yield (a, tuple(rs))


def _inter_cands(views, k, cap, union=False):
    """k held-out in-branches of a common tail -> k-i (or 2u for k=2,union) query."""
    Hin = views.heldout_in
    tails = list(Hin.keys())
    random.shuffle(tails)
    n = 0
    for t in tails:
        branches = [(h, r) for r, hs in Hin[t].items() for h in hs]
        if len(branches) < k:
            continue
        random.shuffle(branches)
        # a handful of k-subsets per tail for diversity without C(d,k) blow-up
        pool = branches[:min(len(branches), k + 6)]
        import itertools
        for combo in itertools.combinations(pool, k):
            atoms = tuple((h, (r,)) for (h, r) in combo)
            yield atoms + ((UNION,),) if union else atoms
            n += 1
            if n >= cap:
                return


def _pi_cands(views, cap):
    """held-out 2-path (a1,(r1,r2)) + held-out in-branch (a2,r3) of a common answer."""
    Hin = views.heldout_in
    n = 0
    for a1, rs, reach in _ho_paths(views, 2, cap):
        r1, r2 = rs
        for t in list(reach):
            din = Hin.get(t)
            if not din:
                continue
            for r3, hs in din.items():
                for a2 in hs:
                    yield ((a1, (r1, r2)), (a2, (r3,)))
                    n += 1
                    if n >= cap:
                        return
                    break  # one anchor per (t, r3) is enough for diversity
                break      # one relation per t


def _ip_cands(views, cap, union=False):
    """two held-out in-branches of an intermediate v + held-out projection r3 from v."""
    H = views.heldout_out
    Hin = views.heldout_in
    inter = list(Hin.keys())
    random.shuffle(inter)
    n = 0
    import itertools
    for v in inter:
        branches = [(h, r) for r, hs in Hin[v].items() for h in hs]
        hv = H.get(v)
        if len(branches) < 2 or not hv:
            continue
        random.shuffle(branches)
        r3s = list(hv.keys())
        for (b1, b2) in itertools.combinations(branches[:6], 2):
            for r3 in r3s[:3]:
                inner = ((b1[0], (b1[1],)), (b2[0], (b2[1],)))
                if union:
                    inner = inner + ((UNION,),)
                yield (inner, (r3,))
                n += 1
                if n >= cap:
                    return


_GEN = {
    "2p": lambda v, cap: _path_cands(v, 2, cap),
    "3p": lambda v, cap: _path_cands(v, 3, cap),
    "4p": lambda v, cap: _path_cands(v, 4, cap),
    "2i": lambda v, cap: _inter_cands(v, 2, cap),
    "3i": lambda v, cap: _inter_cands(v, 3, cap),
    "4i": lambda v, cap: _inter_cands(v, 4, cap),
    "pi": lambda v, cap: _pi_cands(v, cap),
    "ip": lambda v, cap: _ip_cands(v, cap),
    "2u-DNF": lambda v, cap: _inter_cands(v, 2, cap, union=True),
    "up-DNF": lambda v, cap: _ip_cands(v, cap, union=True),
}


def can_enumerate(type_name):
    return type_name in _GEN


def enumerate_full(type_name, views, N, freq_cap_pct=20.0, cand_cap=None, log=True):
    """Return {query: set(full-inference answers)} with total pairs up to N, under the
    relation/anchor frequency cap. Classifier defines 'full'."""
    canon_rel = _canon_rel(views.inverse, views.num_rel)
    universe = views.universe
    if cand_cap is None:
        cand_cap = max(200000, 60 * N)
    out = defaultdict(set)
    collected = 0
    seen = set()
    rels_freq, anch_freq = {}, {}
    considered = 0
    for grounded in _GEN[type_name](views, cand_cap):
        if collected >= N:
            break
        qt = list2tuple(grounded)
        if qt in seen:
            continue
        seen.add(qt)
        considered += 1
        train_set = achieve_answer(_t2l(qt), views.observed_in, views.observed_out, universe)
        n_full, full_set, partials, rels, anchs = reduction.compute_fi_pi_answers(
            type_name, qt, views, train_set, set())
        pool = full_set - out.get(qt, set())
        if not pool:
            continue
        take = min(len(pool), N - collected)
        chosen = pool if take >= len(pool) else set(random.sample(list(pool), take))
        rtmp, atmp, new_rel = add_to_freq_dict(len(chosen), set(rels), set(anchs),
                                               dict(rels_freq), dict(anch_freq), canon_rel)
        if collected > N / 10:
            tkr, tvr = top_k_dict_values_sorting(rtmp, 1)
            tka, tva = top_k_dict_values_sorting(atmp, 1)
            denom = collected + len(chosen)
            if (tvr[0] * 100 / denom >= freq_cap_pct and tkr[0] in new_rel) or \
               (tva[0] * 100 / denom >= freq_cap_pct and tka[0] in anchs):
                continue
        rels_freq, anch_freq = rtmp, atmp
        out[qt] |= chosen
        collected += len(chosen)
    if log:
        logging.info("  enum-full %-10s %6d/%d pairs, %5d queries (%d cands considered)%s",
                     type_name, collected, N, len(out), considered,
                     "  <SHORTFALL>" if collected < N else "")
    return out, collected
