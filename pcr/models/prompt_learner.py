"""Per-identity, per-body-part prompt learning, generalized from CLIP-ReID's PromptLearner
(../CLIP-ReID/model/make_model_clipreid.py, lines 191-239), with a relational-mixing step
(TextualAttentionBlock, pcr/models/relation_blocks.py) across the K part branches' context tokens
before any part's prompt is assembled -- see progress.md's entry on this change for why.

Branch 0 (foreground, matching BPBreIDEncoder.forward's own convention) stays completely outside
the relational-mixing step: it keeps its own independent learnable context, exactly as before
this file gained a TextualAttentionBlock. Branches 1..K (the K parts) share one flat context sequence
that TextualAttentionBlock mixes together, then this class slices back into K per-part 4-token
blocks. See pcr/models/relation_blocks.py's own module docstring for why foreground is excluded
(the source plan frames it as an optional addition, not the base case).
"""
import clip
import torch
import torch.nn as nn

from .relation_blocks import TextualAttentionBlock


class PromptLearner(nn.Module):
    """n_ctx serves two purposes that CLIP-ReID's original code keeps as two separately-named
    variables (n_ctx, n_cls_ctx) always set to the same literal 4: the number of "X" placeholder
    tokens in the fixed template (which determines where the frozen prefix ends and suffix
    begins), and the length of the learnable context spliced into their place. One name here
    instead of two, since they must always be equal for the prefix/suffix slicing to line up.

    Two separate learnable context tensors, not one: `fg_ctx` (foreground, [num_identities,
    n_ctx, ctx_dim], never touched by TextualAttentionBlock) and `part_ctx` (the K parts, flat as
    [num_identities, K*n_ctx, ctx_dim] so TextualAttentionBlock can attend across all of them at
    once -- a transformer layer needs its input as one sequence, not K separate blocks).

    Deviation from CLIP-ReID's original, found by actually running the training loop (back when
    this repo still only had UDA/USL, long before this file existed): CLIP-ReID creates its
    context directly in clip_model.dtype (fp16 on GPU). `torch.amp.GradScaler.step()` hard-errors
    ("Attempting to unscale FP16 gradients") on an fp16 leaf parameter -- mixed-precision
    training requires fp32 master weights. Fixed here: both context tensors are fp32 always, cast
    to the frozen buffers' dtype only inside build_part_prompts() when assembling each prompt --
    autograd handles the cast's backward correctly (the incoming gradient is cast back to fp32
    for accumulation into the fp32 parameter).
    """

    def __init__(self, num_identities, num_parts, clip_text_encoder, n_ctx=4,
                 tab_num_heads=4, tab_num_layers=1, device='cuda'):
        super(PromptLearner, self).__init__()
        # transformer_width, NOT embed_dim -- these tokens get concatenated with token_prefix/
        # token_suffix below (built from clip_text_encoder.token_embedding, which outputs
        # transformer_width-sized vectors) and fed through the transformer itself, which also
        # operates at transformer_width throughout; embed_dim (the *final*, post-text_projection
        # size that must match BPBreID's dim_reduce_output) only applies to ClipTextEncoder's own
        # output, after this class is done. See clip_text_encoder.py's own docstring -- ViT-B/16
        # has transformer_width == embed_dim (both 512), which is why using embed_dim here never
        # broke anything until this repo switched to RN50 (512 vs. 1024).
        ctx_dim = clip_text_encoder.transformer_width
        dtype = clip_text_encoder.dtype

        ctx_init = "A photo of a " + " ".join(["X"] * n_ctx) + " person."
        tokenized_prompts = clip.tokenize(ctx_init).to(device)
        with torch.no_grad():
            embedding = clip_text_encoder.token_embedding(tokenized_prompts).type(dtype)

        # fp32 master weights, cast to the frozen buffers' dtype only in build_part_prompts() --
        # see the class docstring for why (GradScaler forbids fp16 leaf parameters)
        fg_vectors = torch.empty(num_identities, n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(fg_vectors, std=0.02)
        self.fg_ctx = nn.Parameter(fg_vectors)

        part_vectors = torch.empty(num_identities, num_parts * n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(part_vectors, std=0.02)
        self.part_ctx = nn.Parameter(part_vectors)

        self.tab = TextualAttentionBlock(ctx_dim, n_ctx=n_ctx, num_heads=tab_num_heads,
                                          num_layers=tab_num_layers)
        self.prompt_dtype = dtype

        # not trained, but must move with the module (.cuda()/.to()) -- registered as buffers
        # rather than plain attributes, unlike CLIP-ReID's own hardcoded .cuda() call
        self.register_buffer('tokenized_prompts', tokenized_prompts)  # [1, 77]
        self.register_buffer('token_prefix', embedding[:, :n_ctx + 1, :])
        self.register_buffer('token_suffix', embedding[:, n_ctx + 1 + n_ctx:, :])

        self.num_identities = num_identities
        self.num_parts = num_parts
        self.num_branches = 1 + num_parts
        self.n_ctx = n_ctx

    def _splice(self, ctx):
        """ctx: [B, n_ctx, ctx_dim], already dtype-cast. Returns [B, 77, ctx_dim]: frozen prefix
        + ctx + frozen suffix."""
        b = ctx.size(0)
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=1)

    def build_part_prompts(self, labels, part_visibility):
        """labels: [B] identity indices. part_visibility: [B, num_parts], each identity's mean
        per-part visibility (see examples/train_relational_prompts.py's
        compute_identity_visibility and relation_blocks.py's own docstring for why this is a
        per-identity, not per-image, signal) -- passed straight through to TextualAttentionBlock
        as a soft attention bias. Returns a list of `num_branches` tensors, each [B, 77, ctx_dim]
        -- index 0 is the foreground prompt (built from the unmixed fg_ctx), indices 1..K are the
        K part prompts (built from part_ctx after one shared TextualAttentionBlock pass mixes all
        K parts' context together, then sliced back apart)."""
        fg = self._splice(self.fg_ctx[labels].type(self.prompt_dtype))

        raw_part_ctx = self.part_ctx[labels]  # [B, K*n_ctx, ctx_dim], fp32 -- matches tab's fp32 params
        mixed_part_ctx = self.tab(raw_part_ctx, part_visibility).type(self.prompt_dtype)  # cast after tab, like fg
        parts = []
        for k in range(self.num_parts):
            start = k * self.n_ctx
            parts.append(self._splice(mixed_part_ctx[:, start:start + self.n_ctx, :]))

        return [fg] + parts
