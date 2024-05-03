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


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class ErParabolic(ContinualModel):
    NAME = 'er_parabolic'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErParabolic, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device, mode=self.args.buffer_mode)
        
    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        batch_size = real_batch_size
        self.opt.zero_grad()
        # get bridges
        if self.buffer.is_empty():
            mix_inputs, mix_labels = get_bridges(self.args, inputs, labels, 
                                                self.net.num_classes, self.device)
        else:
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            labels = torch.cat((labels, buf_labels))
            mix_inputs, mix_labels = get_bridges(self.args, inputs, labels, 
                                                self.net.num_classes, self.device)
            batch_size += buf_inputs.shape[0]
        # feed forward
        mix_inputs.requires_grad = True
        outputs = self.net(mix_inputs)
        
        if self.args.weight:
            # importance weighting scheme
            # integrate loss over bridges
            loss_path = self.loss(outputs, mix_labels).reshape(batch_size, self.args.n_t).sum(1)
            sample_grads = torch.autograd.grad(loss_path.mean(), mix_inputs, 
                                            retain_graph=True, create_graph=True)[0]
            weight = self.importance_weights(mix_inputs, sample_grads, self.args.n_t, batch_size).sum(1)
            
            loss_path = self.loss(outputs, mix_labels).reshape(batch_size, self.args.n_t).sum(1)
            loss = (loss_path - weight).mean()
        else:
            loss = self.loss(outputs, mix_labels)
            
        loss.backward()
        self.opt.step()
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])
        return loss.item()
    
    def importance_weights(self, x, sample_grads, n_t, batch_size):
            """Calcualte importance weight by integrating the sample_grads according to girsanov"""
            weight = (sample_grads * x).reshape(batch_size, n_t, -1).sum(-1) - \
                    0.5*(sample_grads**2).reshape(batch_size, n_t, -1).sum(-1)
            return weight
        