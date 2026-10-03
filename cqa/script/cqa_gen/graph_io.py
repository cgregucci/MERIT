"""Graph I/O: build observed/held-out/full adjacency views + the candidate universe.

Triple files are already-indexed, already-symmetrized integers (`h r t` per line;
tab- or space-separated). We read them verbatim (no inverse-augmentation) so the
observed graph is edge-for-edge identical to ULTRA's conditioning graph.
"""
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass


def construct_graph(paths):
    """Read int triple files into nested adjacency dicts (ent_out/ent_in).

    Returns (ent_in, ent_out, nodes, max_rel) where ent_out[h][r] = {t...},
    ent_in[t][r] = {h...} as defaultdicts (read access auto-creates empty sets).
    """
    ent_in = defaultdict(lambda: defaultdict(set))
    ent_out = defaultdict(lambda: defaultdict(set))
    nodes = set()
    max_rel = -1
    for path in paths:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                h, r, t = int(parts[0]), int(parts[1]), int(parts[2])
                ent_out[h][r].add(t)
                ent_in[t][r].add(h)
                nodes.add(h); nodes.add(t)
                if r > max_rel:
                    max_rel = r
    return ent_in, ent_out, nodes, max_rel


@dataclass
class GraphViews:
    observed_in: dict
    observed_out: dict
    heldout_in: dict
    heldout_out: dict
    full_in: dict
    full_out: dict
    universe: set        # candidate-answer / negation-complement node set
    num_rel: int         # max rel id + 1 over the full graph
    inverse: str         # "plus_one" | "plus_half"


def load_graph_views(spec, base_dir):
    """Build all adjacency views + universe for one split, per its GraphSpec."""
    def fp(files):
        return [os.path.join(base_dir, f) for f in files]

    obs_in, obs_out, obs_nodes, obs_maxr = construct_graph(fp(spec.observed_files))
    held_in, held_out, held_nodes, held_maxr = construct_graph(fp(spec.heldout_files))
    full_in, full_out, full_nodes, full_maxr = construct_graph(
        fp(tuple(spec.observed_files) + tuple(spec.heldout_files)))

    num_rel = full_maxr + 1

    if spec.universe_mode == "all":
        # transductive: all entities (from id2ent.pkl)
        with open(os.path.join(base_dir, "id2ent.pkl"), "rb") as fin:
            id2ent = pickle.load(fin)
        universe = set(range(len(id2ent)))
    elif spec.universe_mode == "observed":
        # inductive: restrict_nodes = nodes of the conditioning (observed) graph
        universe = set(obs_nodes)
    else:
        raise ValueError(spec.universe_mode)

    return GraphViews(obs_in, obs_out, held_in, held_out, full_in, full_out,
                      universe, num_rel, spec.inverse)
