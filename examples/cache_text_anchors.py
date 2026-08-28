"""Runs once, after examples/train_relational_prompts.py finishes: loads Stage 1's trained
PromptLearner (which owns TextualAttentionBlock as a submodule) and its saved
identity_visibility.pth (TextualAttentionBlock's per-identity attention bias -- see
relation_blocks.py's own docstring and train_relational_prompts.py's compute_identity_visibility;
reused unchanged, not recomputed, so this script's prompts match training exactly), builds every
identity's per-branch text embedding, and saves the frozen [num_identities, num_branches,
embed_dim] lookup table Stage 2 (examples/train_relational_finetune.py) actually reads.

Why this is a separate script, not inlined at the end of train_relational_prompts.py the way the
pre-relational-attention version of this pipeline did it: PromptLearner and TextualAttentionBlock are
both frozen and fully done training the moment Stage 1 ends -- their output for any given
identity is now a fixed, deterministic function of that identity's id alone. Recomputing that
forward pass inside every Stage-2 training step would waste compute on something that never
changes. Running it once here and saving the result makes Stage 2 a pure supervised-training loop
against a fixed table, and makes it obvious in the codebase that PromptLearner/TextualAttentionBlock's
useful life ends exactly here -- nothing after this script ever loads prompt_learner.pth again.
"""
from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import torch

from pcr import datasets
from pcr.models.clip_text_encoder import ClipTextEncoder
from pcr.models.prompt_learner import PromptLearner
from pcr.utils.config import load_yaml_config
from pcr.utils.serialization import load_checkpoint


def get_data(name, data_dir):
    return datasets.create(name, osp.join(data_dir, name))


def compute_text_prototypes(prompt_learner, text_encoder, num_identities, num_branches, id_batch,
                             identity_visibility):
    """identity_visibility: [num_identities, num_branches], the exact table
    examples/train_relational_prompts.py computed and saved -- reused unchanged (not
    recomputed) so TextualAttentionBlock's attention bias here matches training exactly, keeping
    part_ctx/TAB's output a deterministic function of identity alone (see relation_blocks.py's own
    docstring)."""
    prompt_learner.eval()
    text_prototypes = torch.zeros(num_identities, num_branches, text_encoder.embed_dim,
                                   dtype=torch.float32, device='cuda')
    with torch.no_grad():
        for start in range(0, num_identities, id_batch):
            ids = torch.arange(start, min(start + id_batch, num_identities), device='cuda')
            part_vis = identity_visibility[ids, 1:]  # [b, num_parts], same slice as training
            prompts = prompt_learner.build_part_prompts(ids, part_vis)  # list of num_branches tensors
            for branch, prompt in enumerate(prompts):
                text_feat = text_encoder(prompt, prompt_learner.tokenized_prompts)
                text_prototypes[ids, branch] = text_feat.float()
    return text_prototypes


def main():
    parser = argparse.ArgumentParser(
        description="Build Stage 2's frozen text-prototype table from a trained Stage-1 checkpoint")
    parser.add_argument('--config', type=str, required=True, metavar='PATH',
                         help="the same config Stage 1 was trained with -- reads its "
                              "logging.logs_dir for prompt_learner.pth and writes "
                              "text_prototypes.pth there too")
    args = parser.parse_args()
    cfg = load_yaml_config(args.config)

    dataset = get_data(cfg.data.dataset, cfg.data.data_dir)
    num_identities = dataset.num_train_pids
    num_parts = cfg.model.parts_num
    num_branches = 1 + num_parts

    text_encoder = ClipTextEncoder(clip_arch=cfg.clip.arch, device='cuda').cuda()
    prompt_learner = PromptLearner(num_identities, num_parts, text_encoder, n_ctx=cfg.clip.n_ctx,
                                    tab_num_heads=cfg.tab.num_heads, tab_num_layers=cfg.tab.num_layers,
                                    device='cuda').cuda()
    prompt_learner_path = osp.join(cfg.logging.logs_dir, 'prompt_learner.pth')
    prompt_learner.load_state_dict(load_checkpoint(prompt_learner_path))
    print('==> Loaded {}'.format(prompt_learner_path))

    identity_visibility_path = osp.join(cfg.logging.logs_dir, 'identity_visibility.pth')
    identity_visibility = load_checkpoint(identity_visibility_path).cuda()

    print('==> Building text-prototype table for {} identities, {} branches'.format(
        num_identities, num_branches))
    text_prototypes = compute_text_prototypes(prompt_learner, text_encoder, num_identities,
                                               num_branches, cfg.data.cache_batch_size,
                                               identity_visibility)

    out_path = osp.join(cfg.logging.logs_dir, 'text_prototypes.pth')
    torch.save({'text_prototypes': text_prototypes.cpu(), 'num_identities': num_identities,
                'num_branches': num_branches}, out_path)
    print('==> Saved {}'.format(out_path))


if __name__ == '__main__':
    main()
