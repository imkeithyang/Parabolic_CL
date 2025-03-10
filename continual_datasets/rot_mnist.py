# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from backbone.MNISTMLP import MNISTMLP

from continual_datasets.perm_mnist import store_mnist_loaders
from continual_datasets.transforms.rotation import Rotation
from continual_datasets.utils.continual_dataset import ContinualDataset


class RotatedMNIST(ContinualDataset):
    NAME = 'rot-mnist'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 10
    N_TASKS = 20

    def get_data_loaders(self):
        transform = transforms.Compose((Rotation(), transforms.ToTensor()))
        train, test = store_mnist_loaders(transform, self)
        return train, test

    @staticmethod
    def get_backbone():
        return MNISTMLP(28 * 28, RotatedMNIST.N_CLASSES_PER_TASK)

    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_loss(self):
        if self.args.buffer_mode == "parabolic":
             loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            loss_fn = F.cross_entropy
        return loss_fn

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_batch_size() -> int:
        return 128

    @staticmethod
    def get_minibatch_size() -> int:
        return RotatedMNIST.get_batch_size()
