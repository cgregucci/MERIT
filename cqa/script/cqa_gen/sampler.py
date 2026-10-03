import random
import time
import logging
from copy import deepcopy
from collections import defaultdict

from . import reduction
from .registry import GEN_NUM_PER_QUERY, TYPE2STRUCT_LIST


def list2tuple(l):
    return tuple(list2tuple(x) if isinstance(x, list) else x for x in l)


def _base_fn(inverse, num_rel):
    """Map a relation to its direction-agnostic base id (for diversity + freq folding)."""
    if inverse == "plus_one":
        return lambda r: r // 2
    half = num_rel // 2
    return lambda r: r % half


def _canon_rel(inverse, num_rel):
    """Canonical (forward) relation id, used as the frequency-dict key."""
    if inverse == "plus_one":
        return lambda r: r - (r % 2)
    half = num_rel // 2
    return lambda r: r if r < half else r - half


def achieve_answer(query, ent_in, ent_out, universe):
    """Reference query executor. Negation complement uses `universe` (a node set)."""
    assert isinstance(query[-1], list)
    all_relation_flag = True
    for ele in query[-1]:
        if (not isinstance(ele, int)) or (ele == -1):
            all_relation_flag = False
            break
    if all_relation_flag:
        if isinstance(query[0], int):
            ent_set = {query[0]}
        else:
            ent_set = achieve_answer(query[0], ent_in, ent_out, universe)
        for i in range(len(query[-1])):
            if query[-1][i] == -2:
                ent_set = set(universe) - ent_set
            else:
                nxt = set()
                for ent in ent_set:
                    nxt |= ent_out[ent][query[-1][i]]
                ent_set = nxt
    else:
        ent_set = achieve_answer(query[0], ent_in, ent_out, universe)
        union_flag = len(query[-1]) == 1 and query[-1][0] == -1
        for i in range(1, len(query)):
            if not union_flag:
                ent_set = ent_set & achieve_answer(query[i], ent_in, ent_out, universe)
            else:
                if i == len(query) - 1:
                    continue
                ent_set = ent_set | achieve_answer(query[i], ent_in, ent_out, universe)
    return ent_set


def fill_query(query_structure, ent_in, ent_out, answer, base_fn):
    """Backward-grounds an empty query structure (mutates it). Returns True if broken."""
    assert isinstance(query_structure[-1], list)
    all_relation_flag = True
    for ele in query_structure[-1]:
        if ele not in ["r", "n"]:
            all_relation_flag = False
            break
    if all_relation_flag:
        r = -1
        for i in range(len(query_structure[-1]))[::-1]:
            if query_structure[-1][i] == "n":
                query_structure[-1][i] = -2
                continue
            found = False
            for j in range(40):
                r_tmp = random.sample(list(ent_in[answer]), 1)[0]
                # accept first relation (r == -1) or any non-inverse of the previous
                if r == -1 or base_fn(r_tmp) != base_fn(r) or r_tmp == r:
                    r = r_tmp
                    found = True
                    break
            if not found:
                return True
            query_structure[-1][i] = r
            answer = random.sample(list(ent_in[answer][r]), 1)[0]
        if query_structure[0] == "e":
            query_structure[0] = answer
        else:
            return fill_query(query_structure[0], ent_in, ent_out, answer, base_fn)
    else:
        same_structure = defaultdict(list)
        for i in range(len(query_structure)):
            same_structure[list2tuple(query_structure[i])].append(i)
        for i in range(len(query_structure)):
            if len(query_structure[i]) == 1 and query_structure[i][0] == "u":
                assert i == len(query_structure) - 1
                query_structure[i][0] = -1
                continue
            broken_flag = fill_query(query_structure[i], ent_in, ent_out, answer, base_fn)
            if broken_flag:
                return True
        for structure in same_structure:
            if len(same_structure[structure]) != 1:
                structure_set = set()
                for i in same_structure[structure]:
                    structure_set.add(list2tuple(query_structure[i]))
                if len(structure_set) < len(same_structure[structure]):
                    return True


def query_rels_anchs(q):
    """Structurally extract (set-of-relation-ids, set-of-anchor-entity-ids) from a grounded
    BetaE query tuple. Mirrors achieve_answer's structure walk; skips -1 (union) / -2 (negation)
    markers. Single source of truth for the frequency cap; equal to
    reduction.compute_fi_pi_answers' rels/anchs."""
    rels, anchs = set(), set()
    last = q[-1]
    all_rel = all(isinstance(e, int) and e != -1 for e in last)
    if all_rel:
        if isinstance(q[0], int):
            anchs.add(q[0])
        else:
            r2, a2 = query_rels_anchs(q[0]); rels |= r2; anchs |= a2
        for e in last:
            if e != -2:
                rels.add(e)
    else:
        union_flag = len(last) == 1 and last[0] == -1
        for i in range(len(q)):
            if union_flag and i == len(q) - 1:
                continue
            r2, a2 = query_rels_anchs(q[i]); rels |= r2; anchs |= a2
    return rels, anchs


def top_k_dict_values_sorting(d, k):
    sorted_items = sorted(d.items(), key=lambda item: item[1], reverse=True)[:k]
    return [i[0] for i in sorted_items], [i[1] for i in sorted_items]


def add_to_freq_dict(n_answers, set_rels, set_anchs, rels_freq, anch_freqs, canon_rel):
    new_set_rel = set()
    for rel in set_rels:
        new_set_rel.add(canon_rel(rel))
    for rel in new_set_rel:
        rels_freq[rel] = rels_freq.get(rel, 0) + n_answers
    for entity in set_anchs:
        anch_freqs[entity] = anch_freqs.get(entity, 0) + n_answers
    return rels_freq, anch_freqs, new_set_rel


def ground_queries(types, views, gen_num, max_ans_num, seed,
                   gen_num_per_query=GEN_NUM_PER_QUERY, max_outer_tries=0,
                   freq_cap_pct=20.0, log_every=0, return_tiers=False):
    """Sample reduction-balanced (q,a) pairs for the given query types.

    `types` is an ordered list of type-name strings. Returns
    (queries, filtered_answers, hard_answers), where
    queries = {struct_tuple: set[grounded_query]}, filtered/hard = {grounded: set}.
    Held-out answer pool = views.heldout_in; fill/answer graph = views.full_*;
    observed graph = views.observed_*; universe = views.universe.

    If `return_tiers=True`, also returns a 4th value `tier_data`:
    `{type_name: {tier_name: {grounded_query: set[answers]}}}` recording which
    sampled (q,a) pairs fall in which reduction tier (the per-tier stratified
    split of `hard_answers`).
    """
    base_fn = _base_fn(views.inverse, views.num_rel)
    canon_rel = _canon_rel(views.inverse, views.num_rel)
    universe = views.universe
    answer_pool = list(views.heldout_in.keys())   # order-preserving (matches dict.keys order)

    queries = defaultdict(set)
    filtered_answers = defaultdict(set)
    hard_answers = defaultdict(set)
    tier_data = defaultdict(lambda: defaultdict(dict))   # type -> tier -> {query: set}
    rel_per_query_overall = {}
    anch_per_query_overall = {}
    random.seed(seed)
    s0 = time.time()

    for type_name in types:
        template = TYPE2STRUCT_LIST[type_name]
        struct_key = list2tuple(template)
        rel_per_query_overall[struct_key] = {}
        anch_per_query_overall[struct_key] = {}
        gen_num_partials = list(gen_num_per_query[type_name])
        full_name = reduction.full_tier_name(type_name)
        partial_names = reduction.reported_partial_names(type_name)
        tot_to_gen = gen_num * (len(gen_num_partials) + 1)
        num_sampled = 0
        tot_qa_pairs = 0
        num_try = 0
        while tot_qa_pairs < tot_to_gen:
            if max_outer_tries and num_try >= max_outer_tries:
                logging.warning("ground_queries[%s]: hit max_outer_tries=%d at "
                                "tot_qa_pairs=%d/%d (shortfall)",
                                type_name, max_outer_tries, tot_qa_pairs, tot_to_gen)
                break
            num_try += 1
            empty = deepcopy(template)
            answer = random.sample(answer_pool, 1)[0]
            broken = fill_query(empty, views.full_in, views.full_out, answer, base_fn)
            if broken:
                continue
            query = empty
            answer_set = achieve_answer(query, views.full_in, views.full_out, universe)
            train_answer_set = achieve_answer(query, views.observed_in, views.observed_out, universe)
            test_answer_set = answer_set - train_answer_set
            if list2tuple(query) in queries[struct_key]:
                continue
            if len(answer_set) == 0:
                continue
            if len(answer_set - train_answer_set) == 0:
                continue
            if "n" in type_name:   # negation types (lowercase 'n'); DNF/DM unions have 'N'
                if len(train_answer_set - answer_set) == 0:
                    continue
            n_sampled_step, full_inf_answers, partial_inf_answers, rels, anchs = \
                reduction.compute_fi_pi_answers(type_name, list2tuple(query), views,
                                                train_answer_set, answer_set)
            if num_sampled < gen_num:
                if n_sampled_step == 0:
                    continue
            if num_sampled + n_sampled_step <= gen_num:
                answerstoadd = set(full_inf_answers)
            else:
                k_to_add = gen_num - num_sampled
                n_sampled_step = k_to_add
                answerstoadd = set(random.sample(list(full_inf_answers), k_to_add))

            full_added = set(answerstoadd)          # full-inference tier contribution
            partial_pieces = []                      # [(gnidx, set)] partial-tier contributions
            copy_gen_num_partials = gen_num_partials.copy()
            for gnidx in range(len(copy_gen_num_partials)):
                if copy_gen_num_partials[gnidx] < gen_num:
                    if copy_gen_num_partials[gnidx] + len(partial_inf_answers[gnidx]) > gen_num:
                        k_to_add = gen_num - copy_gen_num_partials[gnidx]
                        partial_inf_to_add = set(random.sample(list(partial_inf_answers[gnidx]), k_to_add))
                    else:
                        partial_inf_to_add = partial_inf_answers[gnidx]
                    copy_gen_num_partials[gnidx] += len(partial_inf_to_add)
                    answerstoadd = answerstoadd | partial_inf_to_add
                    partial_pieces.append((gnidx, set(partial_inf_to_add)))

            if max(len(answer_set - train_answer_set),
                   len(train_answer_set - answer_set)) > max_ans_num:
                continue
            rel_temp, anch_temp, new_set_rel = add_to_freq_dict(
                len(answerstoadd), set(rels), set(anchs),
                rel_per_query_overall[struct_key].copy(),
                anch_per_query_overall[struct_key].copy(), canon_rel)
            if tot_qa_pairs > (tot_to_gen / 10):
                tk_rel_k, tk_rel_v = top_k_dict_values_sorting(rel_temp, 1)
                perc_rel = (tk_rel_v[0] * 100) / (tot_qa_pairs + len(answerstoadd))
                tk_anch_k, tk_anch_v = top_k_dict_values_sorting(anch_temp, 1)
                perc_anch = (tk_anch_v[0] * 100) / (tot_qa_pairs + len(answerstoadd))
                if (perc_rel >= freq_cap_pct and tk_rel_k[0] in new_set_rel) or \
                   (perc_anch >= freq_cap_pct and tk_anch_k[0] in anchs):
                    continue
            rel_per_query_overall[struct_key] = rel_temp
            anch_per_query_overall[struct_key] = anch_temp

            if len(answerstoadd) > 0:
                qt = list2tuple(query)
                queries[struct_key].add(qt)
                filtered_answers[qt] = answer_set - answerstoadd
                hard_answers[qt] = answerstoadd
                if return_tiers:
                    if full_added:
                        tier_data[type_name][full_name][qt] = full_added
                    for gnidx, piece in partial_pieces:
                        if piece:
                            tier_data[type_name][partial_names[gnidx]][qt] = piece

            num_sampled += n_sampled_step
            tot_qa_pairs += len(answerstoadd)
            gen_num_partials = copy_gen_num_partials
        if log_every:
            logging.info("%s: sampled %d full-inf, %d (q,a) pairs", type_name,
                         num_sampled, tot_qa_pairs)
    if return_tiers:
        return queries, filtered_answers, hard_answers, tier_data
    return queries, filtered_answers, hard_answers
