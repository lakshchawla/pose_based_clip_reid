from __future__ import print_function, absolute_import
import time

import torch

from .utils.meters import AverageMeter


class PCRTrainer_UDA(object):
    """Part-generalized SpCLTrainer_UDA (spcl/trainers.py). Same per-iteration structure
    (DSBN device-count reshape trick splitting source/target 1:1 per batch, joint forward
    pass through the shared encoder, de-arrange, contrastive loss against the hybrid
    memory) -- only difference is `_forward` returns (embeddings, visibility) tuples that
    need the same device-reshape/split treatment applied to both tensors.
    """

    def __init__(self, encoder, memory, source_classes):
        super(PCRTrainer_UDA, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.source_classes = source_classes

    def train(self, epoch, data_loader_source, data_loader_target,
              optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_s = AverageMeter()
        losses_t = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs, s_targets, _ = self._parse_data(source_inputs)
            t_inputs, _, t_indexes = self._parse_data(target_inputs)

            # arrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs.size()

            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs, t_inputs = reshape(s_inputs), reshape(t_inputs)
            inputs = torch.cat((s_inputs, t_inputs), 1).view(-1, C, H, W)

            # forward
            f_out, vis = self._forward(inputs)  # f_out: [N, M, D], vis: [N, M]

            # de-arrange batch
            M, D = f_out.size(1), f_out.size(2)
            f_out = f_out.view(device_num, -1, M, D)
            vis = vis.view(device_num, -1, M)
            f_out_s, f_out_t = f_out.split(f_out.size(1) // 2, dim=1)
            vis_s, vis_t = vis.split(vis.size(1) // 2, dim=1)
            f_out_s = f_out_s.contiguous().view(-1, M, D)
            f_out_t = f_out_t.contiguous().view(-1, M, D)
            vis_s = vis_s.contiguous().view(-1, M)
            vis_t = vis_t.contiguous().view(-1, M)

            # compute loss with the part hybrid memory
            loss_s = self.memory(f_out_s, s_targets, vis_s)
            loss_t = self.memory(f_out_t, t_indexes + self.source_classes, vis_t)

            loss = loss_s + loss_t
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_s.update(loss_s.item())
            losses_t.update(loss_t.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_s {:.3f} ({:.3f})\t'
                      'Loss_t {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_s.val, losses_s.avg,
                              losses_t.val, losses_t.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)
