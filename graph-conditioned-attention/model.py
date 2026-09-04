"""
A graph-to-graph Transformer: attention biased by an explicit graph, refined
layer by layer rather than taken as a fixed input feature.

Core idea: attention is not only a function of token content, but is *biased
by an explicit graph* — and the model's job is to take an initial (possibly
wrong/naive) graph and iteratively refine it into a better one, layer by
layer, with each layer's attention conditioned on the current graph estimate.

Simplifications relative to a competitive graph-conditioned parser: this
version uses a single dense arc-probability matrix as "the graph" (rather
than discrete/relational graph edges with labels), refines it with plain
additive attention bias (rather than learned graph relation embeddings), and
is applied here to unlabeled arc prediction as a proof-of-concept, not a full
transition-based or graph-based parser.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphBiasedAttention(nn.Module):
    """Self-attention whose logits are additively biased by a graph estimate.

    graph_bias: (B, T, T) real-valued -- added directly to attention scores,
    so a strong existing edge (h, d) makes token h attend more to token d
    (and vice versa via a symmetric bias), letting the current graph guide
    where the next layer's attention looks, instead of learning purely from
    content.
    """

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.bias_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, graph_bias, pad_mask=None):
        B, T, C = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, T, T)
        # symmetrized graph bias so both head->dep and dep->head attention are informed
        sym_bias = graph_bias + graph_bias.transpose(-2, -1)
        att = att + self.bias_scale * sym_bias.unsqueeze(1)

        if pad_mask is not None:
            mask = pad_mask[:, None, None, :]
            att = att.masked_fill(~mask, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


class GraphToGraphBlock(nn.Module):
    """One refinement layer: graph-biased attention + FFN, then re-estimates
    the arc-probability matrix from the updated token representations, so
    the NEXT layer sees a refined graph rather than the same initial guess."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = GraphBiasedAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )
        # re-estimates arc logits from (head_repr, dep_repr) pairs via a bilinear map
        self.arc_head = nn.Linear(d_model, d_model)
        self.arc_dep = nn.Linear(d_model, d_model)

    def forward(self, x, graph_bias, pad_mask=None):
        x = x + self.attn(self.ln1(x), graph_bias, pad_mask)
        x = x + self.ffn(self.ln2(x))

        h = self.arc_head(x)
        d = self.arc_dep(x)
        new_graph_logits = h @ d.transpose(-2, -1) / math.sqrt(x.size(-1))  # (B, T, T)
        return x, new_graph_logits


class GraphToGraphTransformer(nn.Module):
    """Stacks GraphToGraphBlocks; each layer refines the graph estimate that
    biases the next layer's attention. Final layer's graph logits are the
    arc predictions."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_len=32, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            [GraphToGraphBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.max_len = max_len

    def forward(self, idx, init_graph, pad_mask=None):
        B, T = idx.shape
        pos_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.emb(idx) + self.pos(pos_ids)

        graph_logits = init_graph  # start from the naive linear-chain graph, as bias (0/1 -> used directly)
        graph_bias = init_graph
        all_graph_logits = []
        for block in self.blocks:
            x, graph_logits = block(x, graph_bias, pad_mask)
            graph_bias = torch.sigmoid(graph_logits)  # refined soft graph feeds into next layer
            all_graph_logits.append(graph_logits)

        return graph_logits, all_graph_logits  # final layer's logits are the prediction


class VanillaArcPredictor(nn.Module):
    """Baseline: plain (non-graph-biased) Transformer encoder, arcs predicted
    once at the end from token representations -- no iterative graph
    refinement, and the initial linear-chain graph is NOT used at all. This
    isolates the effect of graph-conditioned attention."""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_len=32, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)

        def make_block():
            return nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True),
                "ln2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(),
                    nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
                ),
            })

        self.blocks = nn.ModuleList([make_block() for _ in range(n_layers)])
        self.arc_head = nn.Linear(d_model, d_model)
        self.arc_dep = nn.Linear(d_model, d_model)

    def forward(self, idx, init_graph=None, pad_mask=None):
        B, T = idx.shape
        pos_ids = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.emb(idx) + self.pos(pos_ids)

        key_padding_mask = ~pad_mask if pad_mask is not None else None
        for blk in self.blocks:
            xn = blk["ln1"](x)
            attn_out, _ = blk["attn"](xn, xn, xn, key_padding_mask=key_padding_mask, need_weights=False)
            x = x + attn_out
            x = x + blk["ffn"](blk["ln2"](x))

        h = self.arc_head(x)
        d = self.arc_dep(x)
        graph_logits = h @ d.transpose(-2, -1) / math.sqrt(x.size(-1))
        return graph_logits, [graph_logits]


if __name__ == "__main__":
    from data import load_data

    train_ds, dev_ds, vocab = load_data(n_train=20, n_dev=10)
    ids, pad_mask, gold_arcs, init_arcs = train_ds[0]
    ids, pad_mask, init_arcs = ids.unsqueeze(0), pad_mask.unsqueeze(0), init_arcs.unsqueeze(0)

    g2g = GraphToGraphTransformer(vocab.vocab_size, max_len=ids.size(1))
    logits, all_logits = g2g(ids, init_arcs, pad_mask)
    print("g2g logits:", logits.shape, "n layers of refinement:", len(all_logits))

    base = VanillaArcPredictor(vocab.vocab_size, max_len=ids.size(1))
    logits_b, _ = base(ids, init_arcs, pad_mask)
    print("baseline logits:", logits_b.shape)
