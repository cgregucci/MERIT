"""MeritQuery: complex query answering on the MERIT backbone.

UltraQuery's execution engine (postfix stack machine, fuzzy-logic operators,
traversal dropout) is reused unchanged by subclassing; only the relation-projection
hop differs. Each hop runs MERIT's relation transformer on the motif tensor of the
current graph (``kgfm.tasks.build_motif_tensor``) and then the multi-source entity
reasoner.

The backbone is imported from the ``kgfm`` package at the repository root, so both
the root and ``cqa/`` must be on ``PYTHONPATH``.
"""

import types

import ultra.datasets_query  # noqa: F401  (import before ultraquery to break the circular import)
import torch
from torch import nn
from torch.nn import functional as F

from ultra.ultraquery import UltraQuery, SymbolicTraversal

from kgfm.models.merit import RelTransformerMERIT, EntityNBFNetMERIT
from kgfm.tasks import build_motif_tensor


class QueryNBFNetMERIT(EntityNBFNetMERIT):
    """Entity-level reasoner for MeritQuery: the multi-source variant of
    ``EntityNBFNetMERIT`` (as ULTRA's ``QueryNBFNet`` is of ``EntityNBFNet``).

      (1) the initial node features (the fuzzy set spread over the query-relation
          embedding) are given to ``forward``;
      (2) the per-sample query-relation embedding comes from the projection hop;
      (3) it returns a score for every node (sigmoid applied by the caller).

    It has the same parameters as ``EntityNBFNetMERIT``, so pretrained
    ``entity_model.*`` weights load unchanged.
    """

    def bellmanford(self, data, node_features, query, separate_grad=False):
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=query.device)

        hiddens = []
        edge_weights = []
        layer_input = node_features

        for layer in self.layers:
            if separate_grad:
                edge_weight = edge_weight.clone().requires_grad_()

            # multi-source Bellman-Ford: the fuzzy node features are the boundary condition
            hidden = layer(layer_input, query, node_features, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden

        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)  # (bs, num_nodes, input_dim)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {"node_feature": output, "edge_weights": edge_weights}

    def forward(self, data, node_features, relation_representations, query):
        for layer in self.layers:
            layer.relation = relation_representations

        output = self.bellmanford(data, node_features, query)  # (bs, num_nodes, dim)
        score = self.mlp(output["node_feature"]).squeeze(-1)   # (bs, num_nodes)
        return score


class MeritBackbone(nn.Module):
    """MERIT's relation transformer + the multi-source entity reasoner.

    Attribute names match the ``MERIT`` link-prediction checkpoints, so a pretrained
    ``state["model"]`` loads unchanged.
    """

    REL_CLASSES = {
        "RelTransformerMERIT": RelTransformerMERIT,
    }

    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(MeritBackbone, self).__init__()
        rel_cls_name = rel_model_cfg.get("class", "RelTransformerMERIT")
        if rel_cls_name not in self.REL_CLASSES:
            raise ValueError(
                f"unsupported relation_model class {rel_cls_name!r}; "
                f"expected one of {sorted(self.REL_CLASSES)}")
        self.relation_model = self.REL_CLASSES[rel_cls_name](**rel_model_cfg)
        self.entity_model = QueryNBFNetMERIT(**entity_model_cfg)


class RelationProjectionMERIT(nn.Module):
    """Relation-projection hop on the MERIT backbone.

    As ``ultra.ultraquery.RelationProjection``, except that the relation encoder reads
    the motif tensor of the graph. During training the graph changes under traversal
    dropout, so the tensor is rebuilt every hop; at eval it is built once per graph and
    cached on it.
    """

    def __init__(self, model, threshold=0.0):
        super(RelationProjectionMERIT, self).__init__()
        self.model = model
        self.threshold = threshold

    def _motif(self, graph):
        if self.training:
            return build_motif_tensor(graph)
        motif = getattr(graph, "_meritquery_motif", None)
        if motif is None:
            motif = build_motif_tensor(graph)
            graph._meritquery_motif = motif
        return motif

    def forward(self, graph, h_prob, r_index):
        bs = r_index.shape[0]

        motif = self._motif(graph)  # [R, R, 4]
        rel_struct = types.SimpleNamespace(num_nodes=graph.num_relations)
        # relation representations conditioned on the query relation: (bs, num_rel, dim)
        rel_reprs = self.model.relation_model(rel_struct, query=r_index, motif=motif)
        query = rel_reprs[torch.arange(bs, device=r_index.device), r_index]  # (bs, dim)

        # initialize the input with the fuzzy set and query relation
        input = torch.einsum("bn, bd -> bnd", h_prob, query)

        # optional score thresholding against multi-source propagation
        if self.threshold > 0.0:
            temp_prob = h_prob.clone()
            temp_prob[temp_prob <= self.threshold] = 0.0
            input = torch.einsum("bn, bd -> bnd", temp_prob, query)

        output = self.model.entity_model(graph, input, rel_reprs, query)
        return F.sigmoid(output)


class MeritQuery(UltraQuery):
    """UltraQuery on the MERIT backbone.

    Only ``__init__`` (wrap the backbone in ``RelationProjectionMERIT``) and
    ``apply_projection`` (no relation-graph rebuild; the motif tensor is built inside
    the projection) differ from UltraQuery.
    """

    def __init__(self, model, logic="product", dropout_ratio=0.25, threshold=0.0, more_dropout=0.0):
        nn.Module.__init__(self)
        self.model = RelationProjectionMERIT(model, threshold)
        self.symbolic_model = SymbolicTraversal()
        self.logic = logic
        self.dropout_ratio = dropout_ratio
        self.more_dropout = more_dropout

    def apply_projection(self, mask, graph, r_index):
        h_prob = self.stack.pop(mask)
        if self.training:
            sym_h_prob = self.symbolic_stack.pop(mask)
            # apply traversal dropout based on the output of the symbolic model
            graph = self.traversal_dropout(graph, sym_h_prob, r_index)
        else:
            if self.symbolic_traversal:
                sym_h_prob = self.symbolic_stack.pop(mask)

        # detach the variable to stabilize training
        h_prob = h_prob.detach()
        t_prob = self.model(graph, h_prob, r_index)
        self.stack.push(mask, t_prob)
        self.var.push(mask, t_prob)

        if self.symbolic_traversal:
            sym_t_prob = self.symbolic_model(graph, sym_h_prob, r_index)
            self.symbolic_stack.push(mask, sym_t_prob)
            self.symbolic_var.push(mask, sym_t_prob)

        self.IP[mask] += 1
