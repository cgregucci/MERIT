from dataclasses import dataclass


# --------------------------------------------------------------------------- #
# compose primitives                                                          #
# --------------------------------------------------------------------------- #
def _proj_set(view_out, ents, rel):
    """1-hop projection from a *set* of sources over `rel` using `view_out`."""
    out = set()
    for e in ents:
        out |= view_out[e][rel]
    return out


def _path(view_outs, anchor, rels):
    """Chain projection: hop i resolved on view_outs[i]. (== compute_answers_query_Np)."""
    cur = {anchor}
    for vo, r in zip(view_outs, rels):
        cur = _proj_set(vo, cur, r)
    return cur


def _sel(views, bit):
    """bit 0 -> observed graph, bit 1 -> held-out graph."""
    return views.observed_out if bit == 0 else views.heldout_out


# --------------------------------------------------------------------------- #
# EPFO compose: answer set for a given per-atom view assignment (mask)        #
# --------------------------------------------------------------------------- #
def _compose_path(grounded, views, mask):
    anchor, rels = grounded[0], list(grounded[1])
    return _path([_sel(views, b) for b in mask], anchor, rels)


def _compose_inter(grounded, views, mask):
    # grounded = ((e1,(r1,)), (e2,(r2,)), ...) [union marker ignored for 2u]
    out = None
    for i, b in enumerate(mask):
        e, r = grounded[i][0], grounded[i][1][0]
        s = _sel(views, b)[e][r]
        out = set(s) if out is None else (out & s)
    return out if out is not None else set()


def _compose_pi(grounded, views, mask):
    # grounded = ((e1,(r1,r2)), (e2,(r3,)));  mask=(a1,a2,a3) for (r1,r2,r3)
    e1, (r1, r2) = grounded[0][0], grounded[0][1]
    e2, r3 = grounded[1][0], grounded[1][1][0]
    path_ans = _path([_sel(views, mask[0]), _sel(views, mask[1])], e1, [r1, r2])
    branch = _sel(views, mask[2])[e2][r3]
    return path_ans & branch


def _compose_ip(grounded, views, mask):
    # grounded = (((e1,(r1,)),(e2,(r2,))), (r3,));  mask=(a1,a2,a3)
    # (also used for up-DNF: inner intersection then outer projection)
    inner_struct = grounded[0]
    e1, r1 = inner_struct[0][0], inner_struct[0][1][0]
    e2, r2 = inner_struct[1][0], inner_struct[1][1][0]
    r3 = grounded[1][0]
    inner = _sel(views, mask[0])[e1][r1] & _sel(views, mask[1])[e2][r2]
    return _proj_set(_sel(views, mask[2]), inner, r3)


# (type -> (#atoms, compose_fn))
_EPFO_COMPOSE = {
    "2p": (2, _compose_path), "3p": (3, _compose_path), "4p": (4, _compose_path),
    "2i": (2, _compose_inter), "3i": (3, _compose_inter), "4i": (4, _compose_inter),
    "2u-DNF": (2, _compose_inter),          # classifier treats 2u branches as intersection
    "pi": (3, _compose_pi),
    "ip": (3, _compose_ip), "up-DNF": (3, _compose_ip),
}

# Ordered reduction tiers per type: list of (name, [masks], reported).
# masks are bit-tuples in atom order; 0=observed, 1=held-out. The last entry is
# full-inference. "reported" partials feed gen_num_per_query quotas; internal
# subtract-only groups (2u 1p, up 0p) are reported=False.
def _pc_masks(k, pc):
    """all length-k bit tuples with exactly `pc` ones."""
    out = []
    for m in range(1 << k):
        bits = tuple((m >> (k - 1 - i)) & 1 for i in range(k))
        if sum(bits) == pc:
            out.append(bits)
    return out


_EPFO_TIERS = {
    "2p": [("1p", _pc_masks(2, 1), True), ("full", [(1, 1)], True)],
    "3p": [("1p", _pc_masks(3, 1), True), ("2p", _pc_masks(3, 2), True),
           ("full", [(1, 1, 1)], True)],
    "4p": [("1p", _pc_masks(4, 1), True), ("2p", _pc_masks(4, 2), True),
           ("3p", _pc_masks(4, 3), True), ("full", [(1, 1, 1, 1)], True)],
    "2i": [("1p", _pc_masks(2, 1), True), ("full", [(1, 1)], True)],
    "3i": [("1p", _pc_masks(3, 1), True), ("2i", _pc_masks(3, 2), True),
           ("full", [(1, 1, 1)], True)],
    "4i": [("1p", _pc_masks(4, 1), True), ("2i", _pc_masks(4, 2), True),
           ("3i", _pc_masks(4, 3), True), ("full", [(1, 1, 1, 1)], True)],
    # pi atoms (r1,r2 path; r3 branch): 2i = one path + branch; 2p = both path
    "pi": [("1p", [(0, 0, 1), (0, 1, 0), (1, 0, 0)], True),
           ("2i", [(0, 1, 1), (1, 0, 1)], True),
           ("2p", [(1, 1, 0)], True),
           ("full", [(1, 1, 1)], True)],
    # ip atoms (r1,r2 branches; r3 proj): 2i = both branches; 2p = one branch + proj
    "ip": [("1p", [(0, 0, 1), (0, 1, 0), (1, 0, 0)], True),
           ("2i", [(1, 1, 0)], True),
           ("2p", [(1, 0, 1), (0, 1, 1)], True),
           ("full", [(1, 1, 1)], True)],
    # 2u: 1p is subtract-only (not a quota tier)
    "2u-DNF": [("1p", _pc_masks(2, 1), False), ("full", [(1, 1)], True)],
    # up atoms (r1,r2 union branches; r3 proj): 0p subtract-only; reported 1p, 2u
    "up-DNF": [("0p", [(0, 1, 0), (1, 0, 0)], False),
               ("1p", [(0, 0, 1), (1, 0, 1), (0, 1, 1)], True),
               ("2u", [(1, 1, 0)], True),
               ("full", [(1, 1, 1)], True)],
}


@dataclass
class ReductionResult:
    n_full: int
    full_set: set
    partial_tiers: list          # reported partials, shallow->deep
    rels: list
    anchors: list


def _rels_anchors(type_name, grounded):
    """Extract (rels, anchors) per query type."""
    if type_name in ("2p", "3p", "4p"):
        return list(grounded[1]), [grounded[0]]
    if type_name in ("2i", "3i", "4i", "2u-DNF"):
        rels = [b[1][0] for b in grounded if isinstance(b, tuple) and b and b[0] != "u"
                and not (len(b) == 1)]
        # branches are (e,(r,)); union marker ('u',)/(-1,) is skipped
        rels, anchors = [], []
        for b in grounded:
            if isinstance(b, tuple) and len(b) == 2 and not isinstance(b[0], tuple):
                anchors.append(b[0]); rels.append(b[1][0])
        return rels, anchors
    if type_name == "pi":
        e1, (r1, r2) = grounded[0][0], grounded[0][1]
        e2, r3 = grounded[1][0], grounded[1][1][0]
        return [r1, r2, r3], [e1, e2]
    if type_name in ("ip", "up-DNF"):
        inner = grounded[0]
        e1, r1 = inner[0][0], inner[0][1][0]
        e2, r2 = inner[1][0], inner[1][1][0]
        r3 = grounded[1][0]
        return [r1, r2, r3], [e1, e2]
    raise ValueError(type_name)


def compute_reduction_tiers(type_name, grounded, views, train_answer_set):
    """EPFO reduction tiers. Returns ReductionResult (compute-all; no early-stop)."""
    k, compose = _EPFO_COMPOSE[type_name]
    tiers = _EPFO_TIERS[type_name]
    # precompute answers for each mask referenced by the tiers
    needed = {m for _, masks, _ in tiers for m in masks}
    ans = {m: compose(grounded, views, m) for m in needed}

    prior = set()
    reported = []
    full_set = set()
    for name, masks, is_reported in tiers:
        union = set()
        for m in masks:
            union |= ans[m]
        tier_set = union - prior - train_answer_set
        prior |= tier_set
        if name == "full":
            full_set = tier_set
        elif is_reported:
            reported.append(tier_set)
    rels, anchors = _rels_anchors(type_name, grounded)
    return ReductionResult(len(full_set), full_set, reported, rels, anchors)


# --------------------------------------------------------------------------- #
# Negation classifier                                                         #
# pos_exist (partial, reported) vs pos_only_missing (full-inference).         #
# --------------------------------------------------------------------------- #
def compute_negation_tiers(type_name, grounded, views, train_answer_set, answer_set):
    obs, held, full = views.observed_out, views.heldout_out, views.full_out

    if type_name == "2in":
        e1, r1 = grounded[0][0], grounded[0][1][0]
        e2, r2 = grounded[1][0], grounded[1][1][0]
        all_r2 = full[e2][r2]
        train_r1, train_r2 = obs[e1][r1], obs[e2][r2]
        miss_r1 = held[e1][r1]
        new_train = (train_r1 - train_r2) & answer_set
        pos_exist = (train_r1 - all_r2) - new_train
        pos_only = (miss_r1 - all_r2) - pos_exist - new_train
        return ReductionResult(len(pos_only), pos_only, [pos_exist], [r1, r2], [e1, e2])

    if type_name == "3in":
        e1, r1 = grounded[0][0], grounded[0][1][0]
        e2, r2 = grounded[1][0], grounded[1][1][0]
        e3, r3 = grounded[2][0], grounded[2][1][0]
        all_r3 = full[e3][r3]
        new_train = train_answer_set & answer_set
        a01 = (obs[e1][r1] & held[e2][r2]) - all_r3
        a10 = (held[e1][r1] & obs[e2][r2]) - all_r3
        a11 = (held[e1][r1] & held[e2][r2]) - all_r3
        pos_exist = (a01 | a10) - new_train
        pos_only = a11 - pos_exist - new_train
        return ReductionResult(len(pos_only), pos_only, [pos_exist], [r1, r2, r3], [e1, e2, e3])

    if type_name == "pin":
        e1, (r1, r2) = grounded[0][0], grounded[0][1]
        e2, r3 = grounded[1][0], grounded[1][1][0]
        all_r3 = full[e2][r3]
        new_train = train_answer_set & answer_set
        a01 = _path([obs, held], e1, [r1, r2]) - all_r3
        a10 = _path([held, obs], e1, [r1, r2]) - all_r3
        a11 = _path([held, held], e1, [r1, r2]) - all_r3
        pos_exist = (a01 | a10) - new_train
        pos_only = a11 - pos_exist - new_train
        return ReductionResult(len(pos_only), pos_only, [pos_exist], [r1, r2, r3], [e1, e2])

    if type_name == "pni":
        # path branch carries a negation marker: grounded[0][1] = (r1, r2, -2)
        e1 = grounded[0][0]
        r1, r2 = grounded[0][1][0], grounded[0][1][1]
        e2, r3 = grounded[1][0], grounded[1][1][0]
        new_train = train_answer_set & answer_set
        all_2p = _path([full, full], e1, [r1, r2])
        easy_r3, miss_r3 = obs[e2][r3], held[e2][r3]
        pos_exist = (easy_r3 - all_2p) - new_train
        pos_only = (miss_r3 - all_2p) - new_train
        return ReductionResult(len(pos_only), pos_only, [pos_exist], [r1, r2, r3], [e1, e2])

    if type_name == "inp":
        inner = grounded[0]
        e1, r1 = inner[0][0], inner[0][1][0]
        e2, r2 = inner[1][0], inner[1][1][0]
        r3 = grounded[1][0]
        all_r2 = full[e2][r2]
        new_train = train_answer_set & answer_set
        a0x = obs[e1][r1] - all_r2          # positive branch observed
        a1x = held[e1][r1] - all_r2         # positive branch held-out
        a0x0 = _proj_set(obs, a0x, r3); a0x1 = _proj_set(held, a0x, r3)
        a1x0 = _proj_set(obs, a1x, r3); a1x1 = _proj_set(held, a1x, r3)
        pos_exist = (a0x1 | a1x0) - new_train
        pos_only = a1x1 - pos_exist - new_train
        return ReductionResult(len(pos_only), pos_only, [pos_exist], [r1, r2, r3], [e1, e2])

    raise ValueError(type_name)


# --------------------------------------------------------------------------- #
# Dispatcher                                                                  #
# --------------------------------------------------------------------------- #
def compute_fi_pi_answers(type_name, grounded, views, train_answer_set, answer_set):
    if type_name in _EPFO_COMPOSE:
        r = compute_reduction_tiers(type_name, grounded, views, train_answer_set)
    else:
        r = compute_negation_tiers(type_name, grounded, views, train_answer_set, answer_set)
    return r.n_full, r.full_set, r.partial_tiers, r.rels, r.anchors


# --------------------------------------------------------------------------- #
# Tier-name helpers (pure functions of the type name; no per-query work).     #
# Used by the sampler/emit to label each sampled (q,a) with its reduction     #
# tier for the per-tier `test-query-reduction/{type}/{tier}/` stratified      #
# folders. Names follow the is-cqa convention: the full-inference tier folder #
# is named by the (short) type itself, partial tiers by their reduced type.   #
# --------------------------------------------------------------------------- #
def reported_partial_names(type_name):
    """Reported partial-tier names, shallow->deep. Order matches the
    `partial_tiers` list returned by compute_fi_pi_answers AND
    GEN_NUM_PER_QUERY[type_name] (so name[gnidx] labels partial_inf_answers[gnidx])."""
    if type_name in _EPFO_TIERS:
        return [name for (name, _masks, is_rep) in _EPFO_TIERS[type_name]
                if is_rep and name != "full"]
    # negation: 3in/pin/inp have one reported partial; 2in/pni are full-inference only
    return ["pos-exist"] if type_name in ("3in", "pin", "inp") else []


def full_tier_name(type_name):
    """Folder name for the full-inference tier (is-cqa convention)."""
    if type_name in _EPFO_TIERS:
        return type_name[:-4] if type_name.endswith("-DNF") else type_name
    return "pos-only-miss"   # negation full-inference
