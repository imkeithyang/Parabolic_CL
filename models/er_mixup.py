# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from models.utils.brownian_utils import get_bridges
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
import numpy as np


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErMixup(ContinualModel):
    NAME = 'er_mixup'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErMixup, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=self.args.buffer_mode)
        self.lossvalue = None
        self.agg_inputs = None
        self.agg_labels = None
        self.agg_loss = None
    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        mix_loss = None
        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            mixbuf_inputs, mixbuf_labels, mixbuf_labels_a, mixbuf_labels_b = self.mixup_data(buf_inputs, buf_labels, device=buf_inputs.device)
            mixbuf_outputs = self.net(mixbuf_inputs)
            mix_loss = -((mixbuf_labels*mixbuf_outputs).sum(1).exp()/((mixbuf_labels_a*mixbuf_outputs).sum(1).exp()+(mixbuf_labels_b*mixbuf_outputs).sum(1).exp())).log().mean()
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
        outputs = self.net(inputs)
        if mix_loss is None:
            loss = self.loss(outputs, labels.long())
        else:
            loss = self.loss(outputs, labels.long()) + 0.1*mix_loss
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,
                            labels=labels[:real_batch_size])

        return loss.item()

    def mixup_data(self, x, y, device, alpha=1.0, use_cuda=True):
        if alpha > 0.:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.
        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).to(device).int()
        else:
            index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index,:]
        onehot_y = F.one_hot(y, num_classes=self.net.num_classes)
        onehot_y_a, onehot_y_b = onehot_y, onehot_y[index]
        mixed_y = lam * onehot_y_a + (1 - lam) * onehot_y_b
        
        return mixed_x, mixed_y, onehot_y_a, onehot_y_b