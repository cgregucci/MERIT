"""Mask-driven enumeration of EVERY reduction tier (full + partials) for every query type,
from the observed/held-out graphs, so that the rare tiers of the inductive graphs are
filled without random sampling.

For each tier we know its masks (reduction._EPFO_TIERS): bit i = 1 -> atom i must be a
HELD-OUT edge, 0 -> OBSERVED edge. We generate candidate groundings following each mask,
then classify with the classifier (reduction.compute_fi_pi_answers) and distribute
its exact per-tier answer sets. One query populates several tiers, so we generate over the
union of a type's masks and fill all tiers together, under the relation/anchor frequency cap.

Returns {tier_name: {query_tuple: set(answers)}} per type.
"""
import random
import logging
import itertools
from collections import defaultdict

from . import reduction
from .sampler import (achieve_answer, list2tuple, add_to_freq_dict,
                      top_k_dict_values_sorting, _canon_rel)

UNION = -1
NEG = -2


def _t2l(x):
    return [_t2l(e) for e in x] if isinstance(x, tuple) else x


def _sel_out(views, b):
    return views.heldout_out if b else views.observed_out


def _sel_in(views, b):
    return views.heldout_in if b else views.observed_in


# --------------------------------------------------------------------------- #
# masked candidate generators (yield grounded query tuples following a mask)  #
# --------------------------------------------------------------------------- #
def _gen_path(views, mask, cap):
    graphs = [_sel_out(views, b) for b in mask]
    anchors = list(graphs[0].keys())
    random.shuffle(anchors)
    n = 0
    for a in anchors:
        frontier = {(): {a}}
        for i in range(len(mask)):
            nxt = {}
            for rs, nodes in frontier.items():
                rel2set = defaultdict(set)
                for v in nodes:
                    gv = graphs[i].get(v)
                    if gv:
                        for r, ts in gv.items():
                            rel2set[r] |= ts
                for r, s in rel2set.items():
                    nxt[rs + (r,)] = s
            frontier = nxt
        for rs in frontier:
            yield (a, rs)
            n += 1
            if n >= cap:
                return


def _gen_inter(views, mask, cap, union=False):
    k = len(mask)
    # tails present as in-node in every required graph
    tails = set(views.heldout_in.keys())
    if any(b == 0 for b in mask):
        tails |= set(views.observed_in.keys())
    tails = list(tails)
    random.shuffle(tails)
    n = 0
    for t in tails:
        cands = []
        ok = True
        for i in range(k):
            din = _sel_in(views, mask[i]).get(t)
            if not din:
                ok = False
                break
            cands.append([(h, r) for r, hs in din.items() for h in hs])
        if not ok:
            continue
        for c in cands:
            random.shuffle(c)
        # a handful of combos per tail for diversity (avoid full product blow-up)
        reps = min(cap - n, 8)
        seen = set()
        for _ in range(reps * 3):
            combo = tuple(c[random.randrange(len(c))] for c in cands)
            if len(set(combo)) != k or combo in seen:
                continue
            seen.add(combo)
            atoms = tuple((h, (r,)) for (h, r) in combo)
            yield atoms + ((UNION,),) if union else atoms
            n += 1
            if n >= cap or len(seen) >= reps:
                break
        if n >= cap:
            return


def _gen_pi(views, mask, cap):
    # atoms (r1,r2 path via mask[0],mask[1]) reach X; branch (a2,r3 via mask[2]) into X
    g1, g2, g3in = _sel_out(views, mask[0]), _sel_out(views, mask[1]), _sel_in(views, mask[2])
    n = 0
    anchors = list(g1.keys())
    random.shuffle(anchors)
    for a1 in anchors:
        # 2-path relseqs from a1 + reach sets
        step1 = defaultdict(set)
        for r1, s1 in g1.get(a1, {}).items():
            step1[(r1,)] |= s1
        paths = {}
        for (r1,), mid in step1.items():
            for v in mid:
                for r2, s2 in g2.get(v, {}).items():
                    paths.setdefault((r1, r2), set()).update(s2)
        for (r1, r2), reach in paths.items():
            for x in list(reach):
                din = g3in.get(x)
                if not din:
                    continue
                r3 = next(iter(din))
                a2 = next(iter(din[r3]))
                yield ((a1, (r1, r2)), (a2, (r3,)))
                n += 1
                if n >= cap:
                    return
                break


def _gen_ip(views, mask, cap, union=False):
    # branches r1(mask0), r2(mask1) into v; project r3(mask2) from v
    g1in, g2in, g3 = _sel_in(views, mask[0]), _sel_in(views, mask[1]), _sel_out(views, mask[2])
    inter = list(set(g1in.keys()) & set(g2in.keys()) & set(g3.keys()))
    random.shuffle(inter)
    n = 0
    for v in inter:
        b1 = [(h, r) for r, hs in g1in[v].items() for h in hs]
        b2 = [(h, r) for r, hs in g2in[v].items() for h in hs]
        r3s = list(g3[v].keys())
        random.shuffle(b1); random.shuffle(b2); random.shuffle(r3s)
        for (x1, x2, r3) in itertools.product(b1[:4], b2[:4], r3s[:3]):
            if x1 == x2:
                continue
            inner = ((x1[0], (x1[1],)), (x2[0], (x2[1],)))
            if union:
                inner = inner + ((UNION,),)
            yield (inner, (r3,))
            n += 1
            if n >= cap:
                return


# type -> (compose-arity generator, base masks source)
_EPFO_GEN = {
    "2p": lambda v, m, cap: _gen_path(v, m, cap),
    "3p": lambda v, m, cap: _gen_path(v, m, cap),
    "4p": lambda v, m, cap: _gen_path(v, m, cap),
    "2i": lambda v, m, cap: _gen_inter(v, m, cap),
    "3i": lambda v, m, cap: _gen_inter(v, m, cap),
    "4i": lambda v, m, cap: _gen_inter(v, m, cap),
    "pi": lambda v, m, cap: _gen_pi(v, m, cap),
    "ip": lambda v, m, cap: _gen_ip(v, m, cap),
    "2u-DNF": lambda v, m, cap: _gen_inter(v, m, cap, union=True),
    "up-DNF": lambda v, m, cap: _gen_ip(v, m, cap, union=True),
}

# --------------------------------------------------------------------------- #
# negation: positive tree masked (held-out) + grounded negation branch        #
# --------------------------------------------------------------------------- #
def _rand_edge(views):
    """A random (head, rel) with a non-empty out-set (for grounding negation branches)."""
    a = random.choice(views._ho_keys if hasattr(views, "_ho_keys") else list(views.full_out.keys()))
    ra = views.full_out.get(a)
    if not ra:
        return None
    r = random.choice(list(ra.keys()))
    return a, r


def _gen_negation(views, type_name, cap):
    """Yield grounded negation queries whose POSITIVE tree is all held-out (full-inference
    candidates) plus, for 3in/pin/inp, mixed positive trees for the pos-exist partial."""
    H = views.heldout_out
    full_keys = list(views.full_out.keys())

    def rand_branch():
        a = random.choice(full_keys)
        ra = views.full_out.get(a)
        if not ra:
            return None
        return a, random.choice(list(ra.keys()))

    n = 0
    if type_name in ("2in", "pni"):
        # positive = single held-out edge (e,r)->t; other branch grounded
        for a in list(H.keys()):
            for r, ts in list(H.get(a, {}).items()):
                if not ts:
                    continue
                nb = rand_branch()
                if not nb:
                    continue
                e2, r2 = nb
                if type_name == "2in":
                    yield ((a, (r,)), (e2, (r2, NEG)))
                else:  # pni: positive (e2,r3) held-out is `a,r`; negated 2p path
                    nb2 = rand_branch()
                    if not nb2:
                        continue
                    yield ((e2, (r2, nb2[1], NEG)), (a, (r,)))
                n += 1
                if n >= cap:
                    return
    elif type_name in ("3in", "pin", "inp"):
        # positive is a 2-atom tree; generate for masks (1,1) full and (0,1)/(1,0) partial
        for mask in [(1, 1), (0, 1), (1, 0)]:
            g0, g1 = _sel_out(views, mask[0]), _sel_out(views, mask[1])
            if type_name == "3in":     # positive 2i into t
                g0in, g1in = _sel_in(views, mask[0]), _sel_in(views, mask[1])
                tails = list(set(g0in.keys()) & set(g1in.keys()))
                random.shuffle(tails)
                for t in tails:
                    b0 = [(h, r) for r, hs in g0in[t].items() for h in hs]
                    b1 = [(h, r) for r, hs in g1in[t].items() for h in hs]
                    nb = rand_branch()
                    if not b0 or not b1 or not nb:
                        continue
                    x0 = random.choice(b0); x1 = random.choice(b1)
                    if x0 == x1:
                        continue
                    yield ((x0[0], (x0[1],)), (x1[0], (x1[1],)), (nb[0], (nb[1], NEG)))
                    n += 1
                    if n >= cap:
                        return
            elif type_name == "pin":   # positive 2p path (e1,r1,r2)
                anchors = list(g0.keys()); random.shuffle(anchors)
                for a in anchors:
                    for r1, mid in list(g0.get(a, {}).items()):
                        for v in mid:
                            for r2 in g1.get(v, {}):
                                nb = rand_branch()
                                if not nb:
                                    continue
                                yield ((a, (r1, r2)), (nb[0], (nb[1], NEG)))
                                n += 1
                                if n >= cap:
                                    return
                                break
            else:                       # inp: (e1,r1)->v proj r3; branches r1(mask0), r3(mask1)
                anchors = list(g0.keys()); random.shuffle(anchors)
                for a in anchors:
                    for r1, mid in list(g0.get(a, {}).items()):
                        for v in mid:
                            for r3 in g1.get(v, {}):
                                nb = rand_branch()
                                if not nb:
                                    continue
                                yield (((a, (r1,)), (nb[0], (nb[1], NEG))), (r3,))
                                n += 1
                                if n >= cap:
                                    return
                                break


# --------------------------------------------------------------------------- #
# driver                                                                        #
# --------------------------------------------------------------------------- #
def enumerate_all_tiers(type_name, views, target, freq_cap_pct=20.0, cand_cap=None, log=True):
    """Return {tier_name: {query: set(answers)}} with each tier up to `target` pairs, under the
    relation/anchor frequency cap. Classifier defines all tiers; one query fills several."""
    canon_rel = _canon_rel(views.inverse, views.num_rel)
    universe = views.universe
    partial_names = reduction.reported_partial_names(type_name)
    full_name = reduction.full_tier_name(type_name)
    tier_names = partial_names + [full_name]
    out = {tn: defaultdict(set) for tn in tier_names}
    counts = {tn: 0 for tn in tier_names}
    seen = set()
    rels_freq = {tn: {} for tn in tier_names}
    anch_freq = {tn: {} for tn in tier_names}
    if cand_cap is None:
        cand_cap = max(300000, 80 * target)

    is_neg = "n" in type_name
    if is_neg:
        cand_iter = _gen_negation(views, type_name, cand_cap)
    else:
        tiers = reduction._EPFO_TIERS[type_name]
        masks = []
        for _name, ms, _rep in tiers:
            masks.extend(ms)
        def gen():
            per = max(1, cand_cap // max(1, len(masks)))
            for m in masks:
                yield from _EPFO_GEN[type_name](views, tuple(m), per)
        cand_iter = gen()

    def tier_pairs(qt):
        """Classify a grounded query -> {tier_name: answer_set}."""
        train = achieve_answer(_t2l(qt), views.observed_in, views.observed_out, universe)
        full_ans = achieve_answer(_t2l(qt), views.full_in, views.full_out, universe)
        n_full, full_set, partials, rels, anchs = reduction.compute_fi_pi_answers(
            type_name, qt, views, train, full_ans)
        d = {full_name: full_set}
        for i, pn in enumerate(partial_names):
            d[pn] = partials[i] if i < len(partials) else set()
        return d, rels, anchs

    for grounded in cand_iter:
        if all(counts[tn] >= target for tn in tier_names):
            break
        qt = list2tuple(grounded)
        if qt in seen:
            continue
        seen.add(qt)
        try:
            tps, rels, anchs = tier_pairs(qt)
        except Exception:
            continue
        for tn in tier_names:
            if counts[tn] >= target:
                continue
            pool = tps.get(tn, set()) - out[tn].get(qt, set())
            if not pool:
                continue
            take = min(len(pool), target - counts[tn])
            chosen = pool if take >= len(pool) else set(random.sample(list(pool), take))
            rtmp, atmp, new_rel = add_to_freq_dict(len(chosen), set(rels), set(anchs),
                                                   dict(rels_freq[tn]), dict(anch_freq[tn]), canon_rel)
            if counts[tn] > target / 10:
                tkr, tvr = top_k_dict_values_sorting(rtmp, 1)
                tka, tva = top_k_dict_values_sorting(atmp, 1)
                denom = counts[tn] + len(chosen)
                if (tvr[0] * 100 / denom >= freq_cap_pct and tkr[0] in new_rel) or \
                   (tva[0] * 100 / denom >= freq_cap_pct and tka[0] in anchs):
                    continue
            rels_freq[tn], anch_freq[tn] = rtmp, atmp
            out[tn][qt] |= chosen
            counts[tn] += len(chosen)

    if log:
        summary = ", ".join(f"{tn}={counts[tn]}" for tn in tier_names)
        logging.info("  enum %-8s target=%d -> %s", type_name, target, summary)
    return {tn: dict(out[tn]) for tn in tier_names}
