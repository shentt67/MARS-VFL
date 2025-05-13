import os
import random

import numpy as np
import torch
import ast

from datetime import datetime

def raise_dataset_exception():
    raise Exception('Unknown dataset, please implement it.')


def raise_split_exception():
    raise Exception('Unknown split, please implement it.')


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def keep_predict_loss(y_true, y_pred):
    return torch.sum(y_true * y_pred)

def update_top_model_one_batch(optimizer, model, output, batch_target, loss_func, args=None):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    if args!=None and args.dataset in ['KUHAR']:
        for md in model:
            torch.nn.utils.clip_grad_norm_(md.parameters(), args.clip_grad_t)
    optimizer.step()
    return loss

def update_model_one_batch(optimizer, model, output, batch_target, loss_func):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def update_bottom_model_one_batch(optimizer, model, output, batch_target, loss_func, args=None):
    loss = loss_func(output, batch_target)
    optimizer.zero_grad()
    loss.backward()
    if args!=None and args.dataset in ['KUHAR']:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_t)
    optimizer.step()
    return

def get_tensor_size(target_tensor, ratio=1):
    _size = 1
    if isinstance(target_tensor, tuple):
        target_tensor = target_tensor[0][0].squeeze()
    for _dim in target_tensor.shape:
        _size = _size * _dim
    return (_size * 4 / (1024 * 1024)) * ratio # MB

def sparsify(tensor, compress_ratio):
    """
    save the top k% elements with max absolute value
    """
    tensor_flatten = tensor.view(-1)
    elemnum = tensor.numel()
    k = max(1, int(elemnum * compress_ratio))

    values, indices = torch.topk(torch.abs(tensor_flatten), k, sorted=False)
    values = tensor_flatten[indices]
    return values, indices


def desparsify(indices, values, tensor_size):
    """
    recover the values into origin size
    """
    tensor_decompressed = torch.zeros(tensor_size, device=values.device)

    tensor_decompressed[indices] = values
    return tensor_decompressed


def compress(tensor, compress_ratio):
    """
    return values and indices
    """
    if isinstance(tensor, tuple):
        tensor = tensor[0]

    tensor_shape = tensor.shape
    tensor_flatten = tensor.reshape(-1)
    elemnum = tensor.numel()

    # sparse
    values, indices = sparsify(tensor_flatten, compress_ratio)

    tensor_compressed = torch.cat([values, indices], dim=0)

    ctx = (tensor_shape, elemnum)
    return decompress(tensor_compressed, ctx)


def decompress(tensor_compressed, ctx):
    """
    In practical，just need to send embeddings to active client, and recover to the desired sizes.
    Refer the origin C-VFL codes.
    """
    tensor_shape, tensor_size = ctx

    k = tensor_compressed.numel() // 2
    values = tensor_compressed[:k].float()
    indices = tensor_compressed[k:].long()

    tensor_decompressed = desparsify(indices, values, tensor_size)

    tensor_decompressed = tensor_decompressed.view(tensor_shape)
    return tensor_decompressed

def getdevice(args):
    return torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

def parser_add(parser):
    currentDateAndTime = datetime.now()
    parser.add_argument('--data_dir', default='/data/VFLBench/', help='data directory')
    parser.add_argument('--dataset', default='NUSWIDE', help='name of dataset')
    parser.add_argument('--device', default=0, type=int, help='GPU number')
    parser.add_argument('--results_dir', default='/data/vfl_benchmark_log/logs/' + str(currentDateAndTime),
                        help='the result directory')
    parser.add_argument('--seeds', default=100, type=int, nargs='+', help='the seeds for multiple runs')
    parser.add_argument('--seed', default=100, type=int, help='the seeds for single run')# 200, 300,
    parser.add_argument('--epoch', default=100, type=int, help='number of training epoch')
    parser.add_argument('--batch_size', default=32, type=int, help='the batch size')
    parser.add_argument('--client_num', default=2, type=int, help='the number of clients')
    parser.add_argument('--pretrained_checkpoint', default=None, help='the checkpoint file')
    parser.add_argument('--test_checkpoint', default=None)
    parser.add_argument('--lr', default=0.001, type=float, help='the learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--start_epoch', default=0, type=int, help='the epoch number of starting training')
    parser.add_argument('--print_steps', default=10, type=int, help='the print step of training logging')
    parser.add_argument('--early_stop', default=10000, type=int, help='the epochs for early stopping')
    parser.add_argument('--aggregation', default='concat',
                        help='the aggregation strategy of client embeddings')  # sum, mean, concat
    parser.add_argument('--local_iterations', default=5, type=int, help='the number of local iterations')
    parser.add_argument('--method_name', default='base', help='the evaluation method')
    parser.add_argument('--optimizer', default='sgd', help='the type of optimizer') # adam, sgd
    parser.add_argument('--active_client', default=0, type=int, help='the active client number')

    # args for efficiency
    parser.add_argument('--clip_grad_t', default=1, type=float, help='the threshold of grad clip, to avoid grad explosion with local updates')

    # args for robustness
    parser.add_argument('--eval_mode', default='efficiency',
                        help='the mode of evaluation')  # efficiency, robustness, security
    parser.add_argument('--perturb_rate_train', default=1, type=float,
                        help='the perturb rate of train/valid data')
    parser.add_argument('--perturb_rate_test', default=1, type=float, help='the perturb rate of test data')
    # parser.add_argument('--perturb_level', default=0.3, type=float, help='the perturb level')
    parser.add_argument('--perturb_type', default='missing', help='the perturb type')  # missing, corrupted, misaligned
    parser.add_argument('--misaligned_client', default=0, type=int, help='the misaligned client')

    # args for missing features
    parser.add_argument('--p_miss_train', default=0.5, type=float, help='the possibility of miss rate')
    parser.add_argument('--p_miss_test', default=0.5, type=float, help='the possibility of miss rate')


    # args for AugVFL
    parser.add_argument('--balance', default=12, type=float, help='the balance param of loss')

    # args for RealVFL
    parser.add_argument('--anchor_client', default=0, type=int, help='the anchor client')

    # args for CELUVFL
    parser.add_argument('--threshold', default=0.5, help='the threshold for celu')
    parser.add_argument('--num_batch_per_workset', default=2, help='the size of workset')

    # args for C-VFL
    parser.add_argument('--compression_ratio', default=0.3, help='the compression ratio')

    # args for LASER-VFL


    # args for PMC/AMC
    # the bach-size set to 100
    parser.add_argument('--attack_train_epochs_mc', default=25, type=int, help='the epochs of attack training')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    parser.add_argument('--step_gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--n_labeled_per_class', type=int, default=100,
                        help='Number of labeled data')
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--lambda_u', default=50, type=float)
    parser.add_argument('--val_iteration', type=int, default=1024,
                        help='Number of iterations')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--T', default=0.8, type=float)
    parser.add_argument('--k', help='top k accuracy',
                        type=int, default=4)

    # general args for feature inference attack
    parser.add_argument('--attack_target_client', default=0, type=int, help='the target client')

    # args for GRNA
    parser.add_argument('--UnknownVarLambda', default=0.25, type=float, help='the params of var loss')
    parser.add_argument('--attack_train_epochs_grna', default=100, type=int, help='the epochs of attack training')

    # args for mia
    parser.add_argument('--auxiliary_ratio', default=0.1, type=float, help='the ratio of auxiliary data')
    parser.add_argument('--attack_train_epochs_mia', default=100, type=int, help='the epochs of attack training')

    # general args for backdoor attack
    parser.add_argument('--target_label', default=2, type=int, help='the target label for backdoor')
    parser.add_argument('--attack_client', default=1, type=int, help='the adversary client')

    # args for TECB
    parser.add_argument('--poison_num', default=4, type=int, help='the target label for backdoor')
    parser.add_argument('--alpha_delta', type=float, default=0.05, help='uap learning rate decay')
    parser.add_argument('--eps', type=float, default=1, help='uap clamp bound')
    parser.add_argument('--backdoor', type=float, default=50, help='backdoor frequency')
    parser.add_argument('--poison_epochs', type=float, default=80, help='poison frequency')
    parser.add_argument('--corruption_amp', type=float, default=5, help='amplication of corruption')

    # args for LFBA
    parser.add_argument('--poison_dimensions', default=25, type=int, help='the dimensions to be poisoned')
    parser.add_argument('--trigger_feature_clip', default=1, type=float, help='the clip ratio of feature trigger')
    parser.add_argument('--poison_rate', default=0.1, type=float, help='the rate of poison samples')
    parser.add_argument('--select_rate', default=1, type=float)
    parser.add_argument('--random_select', action='store_true')
    parser.add_argument('--select_replace', action='store_true')
    parser.add_argument('--poison_all', action='store_true')
    parser.add_argument('--anchor_idx', default=6337, type=int)

    # general defense params for malicious attack
    # possible defenses on/off paras
    parser.add_argument('--ppdl', help='turn_on_privacy_preserving_deep_learning',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--gc', help='turn_on_gradient_compression',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--lap-noise', help='turn_on_lap_noise',
                        type=ast.literal_eval, default=False)
    parser.add_argument('--multistep_grad', help='turn on multistep-grad',
                        type=ast.literal_eval, default=False)
    # paras about possible defenses
    parser.add_argument('--ppdl-theta-u', help='theta-u parameter for defense privacy-preserving deep learning',
                        type=float, default=0.75)
    parser.add_argument('--gc-preserved-percent', help='preserved-percent parameter for defense gradient compression',
                        type=float, default=0.75)
    parser.add_argument('--noise-scale', help='noise-scale parameter for defense noisy gradients',
                        type=float, default=1e-3)
    parser.add_argument('--multistep_grad_bins', help='number of bins in multistep-grad',
                        type=int, default=6)
    parser.add_argument('--multistep_grad_bound_abs', help='bound of multistep-grad',
                        type=float, default=3e-2)

    return parser