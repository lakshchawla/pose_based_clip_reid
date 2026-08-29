"""Bidirectional relational attention across a person's K body-part tokens, applied on both the
visual side (VisualAttentionBlock, over BPAM's pooled part features) and the text side
(TextualAttentionBlock, over PromptLearner's per-part learnable context tokens) -- see
progress.md's entry on this change for the full design rationale.

Both blocks are now visibility-aware (see changes.md's "Red flag 6" / plans/IMPROVEMENT_PLAN.md section
5 for the full rationale): every image, including fully-occluded ones, still reaches both blocks
unfiltered (the upstream image-level filter, pcr/utils/visibility_filter.py, was removed entirely
-- see progress.md's entry on the visibility-filter-to-weighting refactor), but each block now uses
a per-part reliability score as a soft bias on its own self-attention, so a poorly-visible part's
near-garbage token contributes less as a KEY to every other part's post-attention representation --
closing the contamination gap that loss-level weighting (SupConLoss, CosineAlignLoss,
PartTripletLoss) alone could never reach, since those only discount a part's own direct loss
contribution, never the mixing that already happened upstream of the loss.

VisualAttentionBlock uses each image's own, real per-part visibility (available fresh at every
forward call, in both Stage 1 and Stage 2). TextualAttentionBlock has no such per-image signal --
PromptLearner.part_ctx is indexed by identity alone, with no per-image input at all -- so it uses a
per-identity mean visibility instead (that identity's average reliability for each part, across
every cached training image of that identity), computed once in
examples/train_relational_prompts.py and reused unchanged by examples/cache_text_anchors.py so the
final frozen text prototypes are built with the exact same bias training converged against.

Only foreground/global stays outside both blocks entirely, in this build: relational mixing
covers the K=5 part branches (branches 1..K in BPBreIDEncoder's convention), not the foreground
branch (branch 0). The source plan describes the foreground vector as an *optional* addition to
VAB's input ("the K part vectors... optionally plus the foreground vector") rather than the base
case, so this keeps to the base case -- foreground keeps its own independent embedding and its
own independent prompt, unmixed, exactly as before this file existed.
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


class VisualAttentionBlock(nn.Module):
    """Bidirectional self-attention over the K pooled part-feature vectors (image side).
    Permanent inference-time module: trained in Stage 1 (backbone/BPAM frozen, this is one of
    the few trainable things), then carried over and continues training in Stage 2 (jointly with
    the now-unfrozen backbone) -- never discarded, unlike TextualAttentionBlock.

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

    def forward(self, part_tokens, part_visibility):
        """part_tokens: [B, K, C] pooled part features (K=5 parts, not including foreground).
        part_visibility: [B, K], that same image's own per-part visibility score (same branch
        order as part_tokens) -- used as a soft attention bias so a poorly-visible part's
        near-garbage feature contributes less as a key to every other part's post-attention
        representation. Returns [B, K, C], same shape, relationally mixed and L2-normalized."""
        attn_bias = _visibility_attn_bias(part_visibility, self.num_heads)
        relation_out = self.encoder(part_tokens, mask=attn_bias)
        mixed = part_tokens + torch.tanh(self.gate) * relation_out
        return F.normalize(mixed, p=2, dim=-1)


class TextualAttentionBlock(nn.Module):
    """Bidirectional self-attention over a person's K*n_ctx learnable part-context tokens (text
    side), run before any single part's prompt is assembled -- so a part's context can be
    informed by every other part's, which the frozen CLIP text encoder's own causally-masked
    self-attention can never provide on its own (a token can only attend to earlier tokens there).

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

    def forward(self, ctx_tokens, part_visibility):
        """ctx_tokens: [B, K*n_ctx, ctx_dim] -- one batch's raw per-identity part context, laid
        out as K contiguous n_ctx-token blocks (see PromptLearner.build_part_prompts's own
        slicing). part_visibility: [B, K], that identity's mean per-part visibility across every
        cached training image of that identity (there is no single per-image signal here --
        part_ctx has no per-image input at all, see PromptLearner's own docstring) -- expanded so
        every context token belonging to a given part shares that part's score. Returns the same
        shape, relationally mixed."""
        token_visibility = part_visibility.repeat_interleave(self.n_ctx, dim=1)  # [B, K*n_ctx]
        attn_bias = _visibility_attn_bias(token_visibility, self.num_heads)
        return self.encoder(ctx_tokens, mask=attn_bias)
