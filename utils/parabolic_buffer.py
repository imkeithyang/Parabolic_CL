# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from collections import Counter


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class PBuffer:
    """
    The memory buffer of rehearsal method.
    Enhanced with brownian bridge boundary update
    """

    def __init__(self, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.attributes = ['examples', 'labels', 'logits', 'task_labels','loss']
    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self

    def __len__(self):
        return min(self.num_seen_examples, self.buffer_size)

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor, loss: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                # added torch.inf to deal with loss might be > 0
                setattr(self, attr_str, (torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device) - 1).type(typ))

    def add_data(self, examples, labels=None, logits=None, task_labels=None, loss=None):
        """
        Adds the data to the memory buffer according to the parabolic PDE bounding strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :param loss: tensor containing loss
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels, loss)
        
        # sort all losses
        loss = (loss - loss.mean())/loss.mean()/len(torch.unique(labels))
        self.loss = torch.cat((self.loss, loss.to(self.device)))
        self.loss, indices = torch.sort(self.loss, descending=True)
        # take the first "buffer_size" of the loss
        ind_to_save = indices[:self.buffer_size]
        self.loss = self.loss[:self.buffer_size]
        self.num_seen_examples = min(self.buffer_size, torch.sum(self.loss != -1).cpu().numpy())# loss.shape[0]
        # update other saved parameters if applicable
        self.examples = torch.cat((self.examples, examples.to(self.device)))
        self.examples = self.examples[ind_to_save]
        if labels is not None:
            self.labels = torch.cat((self.labels,labels.to(self.device)))
            self.labels = self.labels[ind_to_save]
            #print(Counter(self.labels.cpu().numpy()))
        if logits is not None:
            self.logits = torch.cat((self.logits,logits.to(self.device)))
            self.logits = self.logits[ind_to_save]
        if task_labels is not None:
            self.task_labels = torch.cat((self.task_labels,task_labels.to(self.device)))
            self.task_labels = self.task_labels[ind_to_save]
                
    def get_data(self, size: int, transform: nn.Module = None, return_index=False) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([ee.cpu() for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        if not return_index:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

    def get_data_by_index(self, indexes, transform: nn.Module = None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: nn.Module = None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None:
            def transform(x): return x
        ret_tuple = (torch.stack([transform(ee.cpu())
                                  for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
