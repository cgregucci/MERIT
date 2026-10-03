"""MERIT — Motif-Enriched Relational Inductive Transformer.

An ULTRA-style foundation model whose relation encoder is a Transformer over relation
tokens. The relational motifs of the graph, a dense tensor `M ∈ ℝ^{R×R×T}` of log1p
co-occurrence counts (`kgfm.tasks.build_motif_tensor`), enter the relation tower as a
soft prior in two places:

  * **token content** — a query-independent summary of each relation's own row `M[r,:]`
    (reduced over partners, self-diagonal excluded), projected by a zero-init linear
    map into the token's leading `role_dim` dims;
  * **attention-logit bias** — the pairwise entry `M[i,j]` as an additive per-head
    bias, `Δ[h,i,j] = Σ_t M[i,j,t]·B[t,h]`, with `B ∈ ℝ^{T×H}` zero-init.

Both terms are zero-init, so at step 0 the model is the plain token Transformer. The
entity tower is ULTRA's NBFNet conditioned on the resulting relation representations;
it builds (or reads the cached) `M` and hands it to the relation tower.

Token layout, 64 dims: `[ role(role_dim) | query flag / RNI ]`.

    x0[:, r, :C]      = role(r)       C = role_dim, identical for every query
    x0[q, q, C:]      = 1.0           the query token's all-ones flag
    x0[q, r, C:]      ~ N(0, 1)       random noise on the other tokens (RNI)
"""

import math
import types

import torch
from torch import nn
from torch.nn import functional as F

from kgfm.tasks import build_motif_tensor
from kgfm.models.ultra import EntityNBFNet

# torch.compile'd flex_attention, shared across instances and datasets (compilation is
# per shape). Imported lazily, only when attn_impl="flex" is used (torch >= 2.5).
_FLEX_COMPILED = None


def _get_flex():
    global _FLEX_COMPILED
    if _FLEX_COMPILED is None:
        from torch.nn.attention.flex_attention import flex_attention
        # static compile + power-of-two bucket padding of the query batch at the call
        # site bounds the number of compiled kernels per dataset.
        _FLEX_COMPILED = torch.compile(flex_attention, dynamic=False)
    return _FLEX_COMPILED


class RelTransformerMERIT(nn.Module):
    """MERIT's relation encoder: a Transformer over relation tokens with motif-derived
    token content and an additive motif attention bias.
    """

    def __init__(self, input_dim, hidden_dims, num_heads=8, ff_dim=256, dropout=0.0,
                 rni=True, num_channels=4, attn_impl="mha",
                 role_features=("max", "logdeg"), role_dim=8, **kwargs):
        super().__init__()
        # the config's own dispatch key is forwarded with the block; anything else is a typo
        kwargs.pop("class", None)
        assert not kwargs, f"unknown relation_model keys: {sorted(kwargs)}"
        # ULTRA config idiom: layer count = len(hidden_dims), constant width
        assert all(d == input_dim for d in hidden_dims)
        assert attn_impl in {"mha", "flex"}, f"unknown attn_impl={attn_impl!r}"

        self.input_dim = input_dim
        self.rni = rni
        self.num_heads = num_heads
        self.num_channels = num_channels
        self.attn_impl = attn_impl

        # a comma-separated string ("max,logdeg") is accepted so it can be a CLI value
        if isinstance(role_features, str):
            role_features = [f for f in role_features.split(",") if f]
        self.role_features = list(role_features)
        self.role_dim = int(role_dim)
        self._role_c_in = len(self.role_features) * num_channels
        # eval-time memo of the row summary (see _role_profile); not part of state_dict
        self._prof_cache = None

        # zero-init motif terms: Δ = 0 and role = 0 at step 0
        self.B_param = nn.Parameter(torch.zeros(num_channels, num_heads))       # [T, H]
        self.role_proj = nn.Linear(self._role_c_in, self.role_dim, bias=False)
        nn.init.zeros_(self.role_proj.weight)

        self.layers = nn.ModuleList()
        for _ in range(len(hidden_dims)):
            self.layers.append(
                nn.TransformerEncoderLayer(
                    d_model=input_dim, nhead=num_heads, dim_feedforward=ff_dim,
                    dropout=dropout, batch_first=True, norm_first=True)
            )

    # ------------------------------------------------------------------ token content

    def _row_summary(self, motif):
        """Query-independent summary of each relation's own motif row.

        motif: [R, R, T], M[r, p, t] = co-occurrence of relations r and p in channel t.
        Reduces over the partner axis, excluding the self-diagonal, and returns
        [R, len(role_features) * T]. Permutation-invariant over partners, so it is
        defined on any unseen graph.
        """
        R, _, T = motif.shape
        off = (~torch.eye(R, dtype=torch.bool, device=motif.device)).unsqueeze(-1)  # [R,R,1]
        feats = []
        for name in self.role_features:
            if name == "max":
                # peak off-diagonal co-occurrence
                feats.append((motif * off).amax(dim=1))                                 # [R, T]
            elif name == "logdeg":
                # log1p of the number of distinct partners per channel
                deg = ((motif > 0) & off).sum(dim=1).to(motif.dtype)                    # [R, T]
                feats.append(torch.log1p(deg))
            else:
                raise ValueError(f"unknown role feature {name!r} (expected max|logdeg)")
        return torch.cat(feats, dim=-1)                                                 # [R, C_in]

    def _role_profile(self, motif):
        """`_row_summary`, memoized at eval on the identity of the motif tensor.

        M is constant within a dataset at eval (cached on the data object by the entity
        tower), so the summary is computed once per dataset. During training M is
        rebuilt every batch, so nothing is cached.
        """
        if self.training:
            return self._row_summary(motif)
        if self._prof_cache is None or self._prof_cache[0] is not motif:
            self._prof_cache = (motif, self._row_summary(motif))
        return self._prof_cache[1]

    # ---------------------------------------------------------------- attention bias

    def _build_attn_bias4d(self, motif):
        """Additive logit bias Δ[h,i,j] = Σ_t M[i,j,t]·B[t,h], as [1, H, R, R]."""
        dpair = torch.einsum("ijt,th->hij", motif, self.B_param)  # [H, R, R]
        return dpair.unsqueeze(0)                                 # [1, H, R, R]

    def _build_attn_mask(self, motif, num_unique, num_relations):
        """The bias as the 3D [U*H, R, R] float mask MultiheadAttention expects."""
        bias4d = self._build_attn_bias4d(motif)
        H, U, R = self.num_heads, num_unique, num_relations
        # 3D attn_mask layout is (batch * num_heads, L, S), batch-outer / head-inner
        return bias4d.expand(U, H, R, R).reshape(U * H, R, R)

    def _flex_sa_block(self, layer, x, bias4d):
        """`layer._sa_block` via flex_attention; the bias is read inside the kernel.

        head_dim 8 is below flex_attention's minimum of 16, so q/k/v are zero-padded to
        16 and the true scale 1/sqrt(head_dim) is passed; the padding contributes zero
        to both Q·K and the output.
        """
        attn = layer.self_attn
        num_unique, num_relations, dim = x.shape
        head_dim = dim // attn.num_heads
        qkv = F.linear(x, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(num_unique, num_relations, attn.num_heads, head_dim).transpose(1, 2)
        k = k.view(num_unique, num_relations, attn.num_heads, head_dim).transpose(1, 2)
        v = v.view(num_unique, num_relations, attn.num_heads, head_dim).transpose(1, 2)
        pad = max(16 - head_dim, 0)
        if pad:
            q, k, v = (F.pad(t, (0, pad)) for t in (q, k, v))
        # pad the query batch to the next power of two (bounded number of static
        # kernels); the padded rows are sliced off below
        bucket = 1 << (num_unique - 1).bit_length()
        pad_rows = bucket - num_unique
        if pad_rows:
            q, k, v = (F.pad(t, (0, 0, 0, 0, 0, 0, 0, pad_rows)) for t in (q, k, v))
        bias3 = bias4d[0]                           # [H, R, R], query-independent

        def score_mod(score, b, h, q_idx, kv_idx):
            return score + bias3[h, q_idx, kv_idx]

        out = _get_flex()(q, k, v, score_mod=score_mod, scale=1.0 / math.sqrt(head_dim))
        if pad_rows:
            out = out[:num_unique]
        if pad:
            out = out[..., :head_dim]
        out = out.transpose(1, 2).reshape(num_unique, num_relations, dim)
        return layer.dropout1(attn.out_proj(out))

    # -------------------------------------------------------------------- forward

    def forward(self, rel_struct, query, motif):
        if self.attn_impl == "flex" and not self.training and query.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                out = self._forward_impl(rel_struct, query, motif)
            return out.float()
        return self._forward_impl(rel_struct, query, motif)

    def _forward_impl(self, rel_struct, query, motif):
        # motif: dense [R, R, T]; rel_struct only carries num_nodes
        device = query.device
        num_relations, _, T = motif.shape
        assert T == self.num_channels, \
            f"motif channels T={T} != num_channels={self.num_channels}"

        role = self.role_proj(self._role_profile(motif))            # [R, C]
        C = self.role_dim
        assert C < self.input_dim, \
            f"role_dim={C} leaves no room for the query flag in input_dim={self.input_dim}"

        unique_query, inverse = torch.unique(query, return_inverse=True)
        num_unique = len(unique_query)
        rows = torch.arange(num_unique, device=device)

        # initial tokens [U, R, D] = [ role content | query flag ]
        B_full = torch.zeros(num_unique, num_relations, self.input_dim, device=device)
        B_full[:, :, :C] = role.unsqueeze(0)
        B_full[rows, unique_query, C:] = 1.0

        x = B_full.clone()
        if self.rni:
            noise = torch.randn(num_unique, num_relations, self.input_dim, device=device)
            noise[:, :, :C] = 0                     # role dims stay exact
            noise[rows, unique_query, :] = 0        # the query token is deterministic
            x = x + noise

        # the bias is loop-invariant: built once, reused by every layer
        impl = self.attn_impl if not self.training else "mha"
        if impl != "mha":
            mask = self._build_attn_bias4d(motif)
        else:
            mask = self._build_attn_mask(motif, num_unique, num_relations)

        for layer in self.layers:
            if impl == "flex":
                x = x + self._flex_sa_block(layer, layer.norm1(x), mask)
            else:
                # run the pre-LN attention step explicitly
                x = x + layer._sa_block(layer.norm1(x), mask, None)
            x = x + layer._ff_block(layer.norm2(x))

        return x[inverse]


class EntityNBFNetMERIT(EntityNBFNet):
    """MERIT's entity tower: ULTRA's NBFNet conditioned on relation representations
    from `RelTransformerMERIT`.

    It builds the motif tensor `M` for the relation tower. Training: rebuilt per batch
    on the graph after easy-edge removal and edge dropout. Eval: read from the
    dataset's pre-computed `motif` cache (or built once) and kept on the data object.
    """

    def forward(self, data, batch, relation_model, relation_hyper_flag=False,
                precomputed_rel_emb=None):
        assert precomputed_rel_emb is None, \
            "MERIT does not support precomputed_rel_emb (relation representations " \
            "depend on the per-graph motif tensor)"

        h_index, t_index, r_index = batch.unbind(-1)
        shape = h_index.shape

        if self.training and not self.synthetic:
            data = self.remove_easy_edges(data, h_index, t_index, r_index)
            if self.drop_edge_rate > 0:
                drop_edge_mask = torch.bernoulli((1 - self.drop_edge_rate) * torch.ones(len(data.edge_type), device=h_index.device)).to(bool)
                data.edge_index, data.edge_type = data.edge_index[:, drop_edge_mask], data.edge_type[drop_edge_mask]
            # M on the conditioning graph the positive edge is removed from
            motif = build_motif_tensor(data)
        else:
            if getattr(data, "_motif", None) is None:
                cached = getattr(data, "motif", None)
                if cached is not None:
                    M = cached.M if hasattr(cached, "M") else cached
                    data._motif = M.to(h_index.device)
                else:
                    data._motif = build_motif_tensor(data)
            motif = data._motif

        if not self.synthetic:
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index, num_direct_rel=data.num_relations // 2)
        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()

        rel_struct = types.SimpleNamespace(num_nodes=data.num_relations)
        relation_representations = relation_model(rel_struct, query=r_index[:, 0], motif=motif)

        self.query = relation_representations
        for layer in self.layers:
            layer.relation = relation_representations

        output = self.bellmanford(data, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"]
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)


class MERIT(nn.Module):
    """Motif-Enriched Relational Inductive Transformer."""

    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(MERIT, self).__init__()
        self.relation_model = RelTransformerMERIT(**rel_model_cfg)
        self.entity_model = EntityNBFNetMERIT(**entity_model_cfg)

    def forward(self, data, batch, precomputed_rel_emb=None):
        score = self.entity_model(data, batch, self.relation_model,
                                  relation_hyper_flag=False,
                                  precomputed_rel_emb=precomputed_rel_emb)
        return score
