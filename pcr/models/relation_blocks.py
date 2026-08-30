"""Bidirectional relational attention across a person's M=1+K branches (global/foreground + K
parts, uniformly), applied on both the visual side (VisualAttentionBlock, over BPBreID's pooled
branch features) and the text side (TextualAttentionBlock, over PromptLearner's per-branch
learnable context tokens).

Both blocks are visibility-aware: every image, including fully-occluded ones, still reaches both
blocks unfiltered, but each block uses a per-branch reliability score as a soft bias on its own
self-attention, so a poorly-visible branch's near-garbage token contributes less as a KEY to every
other branch's post-attention representation -- closing the contamination gap that loss-level
weighting (SupConLoss, CosineAlignLoss, PartTripletLoss) alone could never reach, since those only
discount a branch's own direct loss contribution, never the mixing that already happened upstream
of the loss.

VisualAttentionBlock uses each image's own, real per-branch visibility (available fresh at every
forward call, in both Stage 1 and Stage 2). TextualAttentionBlock has no such per-image signal --
PromptLearner.ctx is indexed by identity alone, with no per-image input at all -- so it uses a
per-identity mean visibility instead (that identity's average reliability for each branch, across
every cached training image of that identity), computed once in
examples/train_relational_prompts.py and reused unchanged by examples/cache_text_anchors.py so the
final frozen text prototypes are built with the exact same bias training converged against.

Global/foreground (branch 0) is mixed in on equal footing with the K parts -- both blocks treat
it as just another branch, so it receives real gradient (via Stage 1's SupCon loss, extended to
cover all M branches) instead of sitting outside training entirely.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch's nn.TransformerEncoderLayer automatically switches to a fused, native fast-path kernel
# (torch._transformer_encoder_layer_fwd) whenever a layer is in eval() mode -- found the hard way,
# by hand: cache_text_anchors.py calls TextualAttentionBlock through PromptLearner.eval(), and with
# this session's real per-identity visibility values (a normal, non-uniform additive attn_mask,
# nothing extreme -- no zeros, no huge magnitudes) that fast path produced NaN for every part
# branch, on every identity, deterministically. Confirmed the mask math itself is correct: the
# exact same tensors run through the same module in .train() mode (fast path never engages there)
# and eval() with a *uniform* mask (no real masking effect) both produce finite output -- only the
# combination of eval() mode and a real, non-uniform mask breaks. Disabling this fast path globally
# is the documented way around it (torch.backends.mha docs); this module is the only place in this
# repo that builds an nn.TransformerEncoder, so there's no other fast-path user to slow down, and
# both blocks here are tiny (K=5 tokens) where the fused kernel's speed advantage is negligible
# next to correctness.
torch.backends.mha.set_fastpath_enabled(False)


def _visibility_attn_bias(visibility, num_heads, eps=1e-6):
    """visibility: [B, L], one reliability score per key token, in (0, 1]. Returns an additive
    attention-score bias [B*num_heads, L, L] suitable for nn.TransformerEncoder's own `mask`
    argument: log(v_j) added to every query's raw attention score for key j, before the softmax --
    the same mechanism nn.TransformerEncoderLayer's own key_padding_mask uses for a hard 0/1 mask
    (log(0) = -inf), generalized here to a continuous score, so an unreliable key contributes less
    regardless of how well it happens to correlate with a query in raw dot-product terms. Identical
    across every query row and attention head: only the key index carries information."""
    B, L = visibility.shape
    log_vis = torch.log(visibility.clamp(min=eps))  # [B, L]
    bias = log_vis.unsqueeze(1).expand(B, L, L)      # [B, L, L], broadcast over queries
    return bias.unsqueeze(1).expand(B, num_heads, L, L).reshape(B * num_heads, L, L)


def _replay_with_attention(encoder, x, mask):
    """Manually replays a norm_first nn.TransformerEncoder's own layers with need_weights=True --
    nn.TransformerEncoderLayer.forward always discards attention weights (need_weights=False,
    hardcoded in its _sa_block). Returns (output, last layer's attention [B, L, L], heads
    averaged) -- both VAB and TAB use single-layer encoders today, so "last layer" is the only
    layer; a deeper stack would only expose its final layer's pattern this way."""
    attn = None
    for layer in encoder.layers:
        normed = layer.norm1(x)
        sa_out, attn = layer.self_attn(normed, normed, normed, attn_mask=mask,
                                        need_weights=True, average_attn_weights=True)
        x = x + layer.dropout1(sa_out)
        x = x + layer._ff_block(layer.norm2(x))
    return x, attn


class VisualAttentionBlock(nn.Module):
    """Bidirectional self-attention over the M=1+K pooled branch features (image side): global/
    foreground and all K parts, mixed uniformly. Permanent inference-time module: trained in
    Stage 1 (backbone/BPAM frozen, this is one of the few trainable things), then carried over and
    continues training in Stage 2 (jointly with the now-unfrozen backbone) -- never discarded,
    unlike TextualAttentionBlock.

    A learned, zero-initialized residual gate keeps this a no-op at initialization
    (`tanh(0) == 0`, so `forward` returns `part_tokens` unchanged the moment training starts) and
    doubles as a training-stability/interpretability device: the converged value of
    `torch.tanh(self.gate)` is a direct read on how much relational mixing training actually
    found useful for this run -- a gate that stays near 0 is a real (negative) result, not a bug.
    The visibility-aware attention bias below doesn't disturb this: at gate=0, `relation_out`'s
    value (masked or not) is multiplied by zero either way.

    Output is L2-normalized before returning (see changes.md's now-resolved entry on this): the
    residual sum above can drift away from unit norm as the gate moves off zero, but every
    consumer of this output (SupConLoss in Stage 1, PartTripletLoss/CosineAlignLoss in Stage 2)
    computes similarity assuming unit-normalized inputs -- matching BPBreIDEncoder's own
    foreground/global embedding, which is already normalized before this block ever sees the part
    embeddings. Normalizing here, once, means every caller gets a consistent invariant rather than
    each loss call site needing to remember it separately. Doesn't change the zero-init no-op
    property: at gate=0 this returns `normalize(part_tokens)`, and `part_tokens` arrives already
    unit-normalized from BPBreIDEncoder, so it's a true no-op (up to floating-point precision),
    not just an approximate one.
    """

    def __init__(self, dim, num_heads=4, num_layers=1, ff_dim=None):
        super(VisualAttentionBlock, self).__init__()
        ff_dim = ff_dim or dim * 2
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.gate = nn.Parameter(torch.zeros(1))
        self.num_heads = num_heads

    def forward(self, branch_tokens, branch_visibility):
        """branch_tokens: [B, M, C] pooled branch features (M=1+K: global/foreground + K parts).
        branch_visibility: [B, M], that same image's own per-branch visibility score (same branch
        order as branch_tokens) -- used as a soft attention bias so a poorly-visible branch's
        near-garbage feature contributes less as a key to every other branch's post-attention
        representation. Returns (mixed [B, M, C], attn [B, M, M]) -- mixed is relationally mixed
        and L2-normalized; attn is this call's own self-attention pattern (see
        train_relational_prompts.py's L_relalign, which is the only current consumer)."""
        attn_bias = _visibility_attn_bias(branch_visibility, self.num_heads)
        relation_out, attn = _replay_with_attention(self.encoder, branch_tokens, attn_bias)
        mixed = branch_tokens + torch.tanh(self.gate) * relation_out
        return F.normalize(mixed, p=2, dim=-1), attn


class TextualAttentionBlock(nn.Module):
    """Bidirectional self-attention over a person's M*n_ctx learnable branch-context tokens (text
    side, M=1+K: global/foreground + K parts), run before any single branch's prompt is assembled
    -- so a branch's context can be informed by every other branch's, which the frozen CLIP text
    encoder's own causally-masked self-attention can never provide on its own (a token can only
    attend to earlier tokens there).

    Training-only: exists solely within Stage 1, alongside PromptLearner. Both are frozen and
    discarded once Stage 1 ends -- cache_text_anchors.py reads their state once to build the
    frozen text-prototype table Stage 2 actually uses, and neither is loaded again afterward.

    No gate, no residual, unlike VisualAttentionBlock: this class's own life ends the moment Stage
    1 does, so there's no "does this stay a no-op at inference" concern to guard against the way
    there is for VAB.
    """

    def __init__(self, ctx_dim, n_ctx, num_heads=4, num_layers=1):
        super(TextualAttentionBlock, self).__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim, nhead=num_heads, dim_feedforward=ctx_dim * 2,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.n_ctx = n_ctx
        self.num_heads = num_heads

    def forward(self, ctx_tokens, branch_visibility):
        """ctx_tokens: [B, M*n_ctx, ctx_dim] -- one batch's raw per-identity branch context, laid
        out as M contiguous n_ctx-token blocks (see PromptLearner.build_part_prompts's own
        slicing). branch_visibility: [B, M], that identity's mean per-branch visibility across
        every cached training image of that identity (there is no single per-image signal here --
        ctx has no per-image input at all, see PromptLearner's own docstring) -- expanded so every
        context token belonging to a given branch shares that branch's score. Returns
        (mixed [B, M*n_ctx, ctx_dim], attn [B, M, M]) -- mixed is the same shape, relationally
        mixed; attn is the raw [B, M*n_ctx, M*n_ctx] self-attention pattern reduced to one M x M
        branch-to-branch summary: summed over each key branch's own n_ctx tokens (each query row
        sums to 1 over the full M*n_ctx keys, so summing -- not averaging -- a key block preserves
        that row's total probability mass), then averaged over each query branch's own n_ctx rows
        (a mean of several valid distributions is itself a valid distribution). Needed as a real
        probability distribution, each row summing to 1, since L_relalign
        (train_relational_prompts.py) feeds this into a KL divergence against VAB's native
        [B, M, M] -- naively averaging over both axes (as a purely-visual heatmap wouldn't need to
        care about) leaves each row summing to 1/n_ctx instead, silently breaking KL's
        non-negativity."""
        M = branch_visibility.size(1)
        token_visibility = branch_visibility.repeat_interleave(self.n_ctx, dim=1)  # [B, M*n_ctx]
        attn_bias = _visibility_attn_bias(token_visibility, self.num_heads)
        mixed, attn_full = _replay_with_attention(self.encoder, ctx_tokens, attn_bias)
        B = attn_full.size(0)
        attn = attn_full.view(B, M, self.n_ctx, M, self.n_ctx).sum(dim=4).mean(dim=2)
        return mixed, attn


class CrossAttentionBlock(nn.Module):
    """Multi-head cross-attention: `query_tokens` attends to `context_tokens` from the other
    modality. Tanh-gated, zero-init residual (starts as an identity function). See
    METHODOLOGY.md's Stage 2 / CAB section for how this is used."""

    def __init__(self, dim, num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, query_tokens, context_tokens):
        """query_tokens/context_tokens: [B, N, D]. Returns (updated_query [B, N, D], attn_weights
        [B, N, N] averaged over heads)."""
        B, N, D = query_tokens.shape
        q = self.q_proj(query_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context_tokens).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-1, -2) / (self.head_dim ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        updated_query = query_tokens + torch.tanh(self.gate) * out
        return updated_query, attn.mean(dim=1)
