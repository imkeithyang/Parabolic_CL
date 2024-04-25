# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from continual_datasets import NAMES as DATASET_NAMES
from models import get_all_models


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')
    
    parser.add_argument('--bridge', action='store_true',
                        help='Brownian bridge interpolation.')
    parser.add_argument('--sigma', type=float, default=0.05, 
                        help='bridge sigma')
    parser.add_argument('--T', type=float, default=1.0, 
                        help='terminal time for bridge')
    parser.add_argument('--n_t', type=int, default=5, 
                        help='number of time observation for bridge')
    parser.add_argument('--n_b', type=int, default=1, 
                        help='number of bridge')
    parser.add_argument('--eps', type=float, default=0.05, 
                        help='determine convergence in loss')
    parser.add_argument('--match', action='store_true', default=False, 
                        help='match by minimizing l2')
    parser.add_argument('--ot_match', action='store_true', default=False, 
                        help='match by minimizing OT EMD')
    parser.add_argument('--simplex', action='store_true', default=False, 
                        help='interpolating y on simplex')
    parser.add_argument('--weight', type=int, default=0, 
                        help='Weighting the bridge observation or not')


    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])
    parser.add_argument('--device', type=str, default='cuda:0')


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='hy190', help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='mammoth', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--buffer_mode', type=str, required=True, 
                        choices=["reservoir", "parabolic"], default="reservoir",
                        help='The mode of the buffer update.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')
