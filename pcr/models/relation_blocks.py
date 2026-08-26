"""Bidirectional relational attention across a person's K body-part tokens, applied on both the
visual side (VisualAttentionBlock, over BPAM's pooled part features) and the text side
(TextualAttentionBlock, over PromptLearner's per-part learnable context tokens) -- see
progress.md's entry on this change for the full design rationale.

Neither block masks anything: this plan's operating assumption (see reid_pipeline_plan.md's
part-relational-attention addendum, section 0.1) is that low-visibility images are filtered out
upstream (pcr/utils/visibility_filter.py) before they ever reach either block, so every one of
the K part tokens handed to either block is assumed reliably present. That is a real, stated
assumption, not something these two classes verify themselves -- they always run "as if" every
part is visible.

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

    Output is L2-normalized before returning (see changes.md's now-resolved entry on this): the
    residual sum above can drift away from unit norm as the gate moves off zero, but every
    consumer of this output (InfoNCELoss in Stage 1, PartTripletLoss/CosineAlignLoss in Stage 2)
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

    def forward(self, part_tokens):
        """part_tokens: [B, K, C] pooled part features (K=5 parts, not including foreground).
        Returns [B, K, C], same shape, relationally mixed and L2-normalized."""
        relation_out = self.encoder(part_tokens)
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

    def __init__(self, ctx_dim, num_heads=4, num_layers=1):
        super(TextualAttentionBlock, self).__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=ctx_dim, nhead=num_heads, dim_feedforward=ctx_dim * 2,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, ctx_tokens):
        """ctx_tokens: [B, K*n_ctx, ctx_dim] -- one batch's raw per-identity part context.
        Returns the same shape, relationally mixed."""
        return self.encoder(ctx_tokens)
