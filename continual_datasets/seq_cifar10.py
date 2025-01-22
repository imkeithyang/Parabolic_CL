# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
from backbone.ResNet18 import resnet18
from PIL import Image
from torchvision.datasets import CIFAR10
import numpy as np
from continual_datasets.seq_tinyimagenet import base_path
from continual_datasets.transforms.denormalization import DeNormalize
from continual_datasets.utils.continual_dataset import (ContinualDataset,
                                              store_masked_loaders)
from continual_datasets.utils.validation import get_train_val

class TCIFAR10(CIFAR10):
    """Workaround to avoid printing the already downloaded messages."""
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.root = root
        super(TCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

class MyCIFAR10(CIFAR10):
    """
    Overrides the CIFAR10 dataset to change the getitem function.
    """
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        super(MyCIFAR10, self).__init__(root, train, transform, target_transform, download=not self._check_integrity())

    def __getitem__(self, index: int) -> Tuple[Image.Image, int, Image.Image]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')
        original_img = img.copy()

        not_aug_img = self.not_aug_transform(original_img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]

        return img, target, not_aug_img


class My_Noise_CIFAR10():
    def __init__(self, root, imbalanced_factor=None, order="normal") -> None:
        self.train_dataset = self.build_dataloader(root, imbalanced_factor, order)
        self.data = self.train_dataset.data
        self.targets = self.train_dataset.targets
        self.train_transforms = transforms.Compose(
                [transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2470, 0.2435, 0.2615))])

        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.data)

    def build_dataloader(self,root, imbalanced_factor=2, order="normal"):
        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615)),
        ])

        train_dataset = CIFAR10(root=root, train=True, download=True, transform=train_transforms)
        num_classes = len(train_dataset.classes)

        # index_to_meta = []
        index_to_train = []
        num_meta_total = 0

        if imbalanced_factor is not None:
            imbalanced_num_list = []
            sample_num = int((len(train_dataset.targets) - num_meta_total) / num_classes)
            for class_index in range(num_classes):
                imbalanced_num = sample_num / (imbalanced_factor ** (class_index / (num_classes - 1)))
                imbalanced_num_list.append(int(imbalanced_num))
            # 
            if order == "reverse":
                imbalanced_num_list.reverse()
                print(imbalanced_num_list)
            elif order == "random":
                np.random.shuffle(imbalanced_num_list)  
                print(imbalanced_num_list)
            else:
                print(imbalanced_num_list)
                print('imbalance_factor', imbalanced_factor)
        else:
            imbalanced_num_list = None

        for class_index in range(num_classes):
            index_to_class = [index for index, label in enumerate(train_dataset.targets) if label == class_index]
            np.random.shuffle(index_to_class)
            index_to_class_for_train = index_to_class

            if imbalanced_num_list is not None:
                index_to_class_for_train = index_to_class_for_train[:imbalanced_num_list[class_index]]
            index_to_train.extend(index_to_class_for_train)

        train_dataset.data = train_dataset.data[index_to_train]
        train_dataset.targets = list(np.array(train_dataset.targets)[index_to_train])
 
        return train_dataset   

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # to return a PIL Image
        # not_aug_img = img
        img = Image.fromarray(img, mode='RGB')
        # original_img = img.copy()
        not_aug_img = self.not_aug_transform(img)

        # if self.train_transforms is not None:
        img = self.train_transforms(img)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        if hasattr(self, 'logits'):
            return img, target, not_aug_img, self.logits[index]
        return img, target, not_aug_img

class SequentialCIFAR10(ContinualDataset):
    # 继承自ContinualDataset, 因此有 self.i = 0
    NAME = 'seq-cifar10'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 5
    task_id = 0
    TRANSFORM = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2470, 0.2435, 0.2615))])


    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = transforms.Compose(
            [transforms.ToTensor(), self.get_normalization_transform()])

        train_dataset = MyCIFAR10(base_path() + 'CIFAR10', train=True,
                                  download=True, transform=transform) 
        if self.args.imbalance:
            train_dataset = My_Noise_CIFAR10(base_path() + 'CIFAR10', order = self.args.order)

        if self.args.validation:
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                    test_transform, self.NAME)
        else:
            test_dataset = CIFAR10(base_path() + 'CIFAR10',train=False,
                                   download=True, transform=test_transform)
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    @staticmethod
    def get_transform():
        transform = transforms.Compose(
            [transforms.ToPILImage(), SequentialCIFAR10.TRANSFORM])
        return transform

    @staticmethod
    def get_backbone():
        return resnet18(SequentialCIFAR10.N_CLASSES_PER_TASK
                        * SequentialCIFAR10.N_TASKS)

    @staticmethod
    def get_loss(args):
        if args.weight or args.model=="er_mv" or args.model=="er_parabolic":
             loss_fn = nn.CrossEntropyLoss(reduction="none")
        else:
            loss_fn = F.cross_entropy
        return loss_fn

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2615))
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return SequentialCIFAR10.get_batch_size()
