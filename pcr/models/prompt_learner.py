"""Per-identity, per-branch prompt learning, generalized from CLIP-ReID's PromptLearner
(../CLIP-ReID/model/make_model_clipreid.py, lines 191-239), with a relational-mixing step
(TextualAttentionBlock, pcr/models/relation_blocks.py) across all M=1+K branches' context tokens
(global/foreground + K parts, uniformly) before any branch's prompt is assembled.
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

    One learnable context tensor, `ctx` ([num_identities, num_branches*n_ctx, ctx_dim]), flat so
    TextualAttentionBlock can attend across all M=1+K branches at once (branch 0 = global/
    foreground, 1..K = parts) -- a transformer layer needs its input as one sequence, not M
    separate blocks.

    Deviation from CLIP-ReID's original, found by actually running the training loop (back when
    this repo still only had UDA/USL, long before this file existed): CLIP-ReID creates its
    context directly in clip_model.dtype (fp16 on GPU). `torch.amp.GradScaler.step()` hard-errors
    ("Attempting to unscale FP16 gradients") on an fp16 leaf parameter -- mixed-precision
    training requires fp32 master weights. Fixed here: `ctx` is fp32 always, cast to the frozen
    buffers' dtype only inside build_part_prompts() when assembling each prompt -- autograd
    handles the cast's backward correctly (the incoming gradient is cast back to fp32 for
    accumulation into the fp32 parameter).
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

        num_branches = 1 + num_parts

        # fp32 master weights, cast to the frozen buffers' dtype only in build_part_prompts() --
        # see the class docstring for why (GradScaler forbids fp16 leaf parameters)
        ctx_vectors = torch.empty(num_identities, num_branches * n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

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
        self.num_branches = num_branches
        self.n_ctx = n_ctx

    def _splice(self, ctx):
        """ctx: [B, n_ctx, ctx_dim], already dtype-cast. Returns [B, 77, ctx_dim]: frozen prefix
        + ctx + frozen suffix."""
        b = ctx.size(0)
        prefix = self.token_prefix.expand(b, -1, -1)
        suffix = self.token_suffix.expand(b, -1, -1)
        return torch.cat([prefix, ctx, suffix], dim=1)

    def build_part_prompts(self, labels, branch_visibility):
        """labels: [B] identity indices. branch_visibility: [B, num_branches], each identity's
        mean per-branch visibility, branch 0 (global/foreground) included (see
        examples/train_relational_prompts.py's compute_identity_visibility) -- passed straight
        through to TextualAttentionBlock as a soft attention bias. Returns (prompts, tab_attn):
        prompts is a list of `num_branches` tensors, each [B, 77, ctx_dim], in branch order (0 =
        global/foreground, 1..K = parts) -- one shared TextualAttentionBlock pass mixes all M
        branches' context together before this class slices back into per-branch n_ctx-token
        blocks. tab_attn is that same call's own [B, num_branches, num_branches] attention
        pattern (see TextualAttentionBlock.forward), most callers ignore it -- only
        train_relational_prompts.py's L_relalign consumes it."""
        raw_ctx = self.ctx[labels]  # [B, num_branches*n_ctx, ctx_dim], fp32 -- matches tab's fp32 params
        mixed_ctx, tab_attn = self.tab(raw_ctx, branch_visibility)
        mixed_ctx = mixed_ctx.type(self.prompt_dtype)
        prompts = []
        for b in range(self.num_branches):
            start = b * self.n_ctx
            prompts.append(self._splice(mixed_ctx[:, start:start + self.n_ctx, :]))

        return prompts, tab_attn
