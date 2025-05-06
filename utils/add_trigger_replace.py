import copy

from utils.utils import *
from torch.nn.utils.rnn import pack_padded_sequence

def add_trigger_to_data_replace(args, logger, replace_indexes_others, replace_indexes_target, train_indexes,
                                poison_indexes, new_data, trigger_dimensions, new_targets, rate, mode,
                                replace_label, length = None):
    mode_print(logger, mode)
    if args.dataset == 'CIFAR10':
        new_data, new_targets = add_triangle_pattern_trigger(args, logger, replace_indexes_others,
                                                             replace_indexes_target, train_indexes, poison_indexes,
                                                             new_data,
                                                             new_targets, rate,
                                                             mode, replace_label)
        return new_data, new_targets
    elif args.dataset == 'UCIHAR':
        new_data, new_targets = add_feature_trigger(args, logger, replace_indexes_others, replace_indexes_target,
                                                    train_indexes, poison_indexes, trigger_dimensions, new_data,
                                                    new_targets, rate, mode,
                                                    replace_label)
        return new_data, new_targets
    elif args.dataset == 'PHISHING':
        new_data, new_targets = add_vector_replacement_trigger(args, logger, replace_indexes_others,
                                                               replace_indexes_target, train_indexes, poison_indexes,
                                                               trigger_dimensions, new_data,
                                                               new_targets, rate, mode,
                                                               replace_label)
        return new_data, new_targets
    elif args.dataset == 'NUSWIDE':
        new_data, new_targets = add_vector_replacement_trigger(args, logger, replace_indexes_others,
                                                               replace_indexes_target, train_indexes, poison_indexes,
                                                               trigger_dimensions, new_data, new_targets,
                                                               rate,
                                                               mode, replace_label)
        return new_data, new_targets


def add_triangle_pattern_trigger(args, logger, replace_indexes_others, replace_indexes_target, train_indexes,
                                 poison_indexes, new_data, new_targets, rate, mode,
                                 replace_label):
    height, width, channels = new_data.shape[1:]
    temp = copy.deepcopy(new_data)
    for i, idx in enumerate(replace_indexes_others):
        for c in range(channels):
            temp[idx, height - 3:, width - 3:, c] = 0
            temp[idx, height - 3, width - 1, c] = 255
            temp[idx, height - 1, width - 3, c] = 255
            temp[idx, height - 2, width - 2, c] = 255
            temp[idx, height - 1, width - 1, c] = 255
        if args.client_num == 2:
            new_data[replace_indexes_target[i], :, 16:, :] = temp[idx, :, 16:, :]
        elif args.client_num == 4:
            new_data[replace_indexes_target[i], 16:, 16:, :] = temp[idx, 16:, 16:, :]
    logger.info(
        "Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def add_feature_trigger(args, logger, replace_indexes_others, replace_indexes_target, train_indexes, poison_indexes,
                        trigger_dimensions, new_data, new_targets,
                        rate, mode,
                        replace_label=True):
    temp = copy.deepcopy(new_data)
    for i, idx in enumerate(replace_indexes_others):
        temp[idx][trigger_dimensions] = args.trigger_feature_clip
        if args.dataset == 'UCIHAR':
            # Acceleration Sensor: 1-120,201-239,266-423,503-528,555-556,559-561 (348 in total)
            # Gyroscope Sensor: 121-200, 240-265, 424-502, 529-554, 557-558 (213 in total)
            acc_ind = list(range(0, 120)) + list(range(200, 239)) + list(range(265, 423)) + list(
                range(502, 528)) + list(range(554, 556)) + list(range(558, 561))
            gyr_ind = list(range(120, 200)) + list(range(239, 265)) + list(range(423, 502)) + list(
                range(528, 554)) + list(range(556, 558))
            candidate = [acc_ind, gyr_ind]
            if args.client_num == 2:
                new_data[replace_indexes_target[i]][candidate[args.attack_client]] = temp[idx][candidate[args.attack_client]]
            if args.client_num == 4:
                new_data[replace_indexes_target[i]][421:] = temp[idx][421:]
        elif args.dataset == 'NUSWIDEI':
            new_data[replace_indexes_target[i]][:634] = temp[idx][:634]
    logger.info(
        "Add Trigger to %d Bad Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def add_vector_replacement_trigger(args, logger, replace_indexes_others, replace_indexes_target, train_indexes,
                                   poison_indexes, trigger_dimensions, new_data,
                                   new_targets, rate, mode, replace_label, length = None):
    temp = copy.deepcopy(new_data)
    if args.dataset == 'PHISHING':
        if args.client_num == 2:
            for i, idx in enumerate(replace_indexes_others):
                temp[idx][trigger_dimensions] = 1
                new_data[replace_indexes_target[i]][15:] = temp[idx][15:]
        elif args.client_num == 4:
            for i, idx in enumerate(replace_indexes_others):
                temp[idx][trigger_dimensions] = 1
                new_data[replace_indexes_target[i]][23:] = temp[idx][23:]
    elif args.dataset == 'NUSWIDE':
        if args.client_num == 2:
            for i, idx in enumerate(replace_indexes_others):
                temp[idx][trigger_dimensions] = 1
                new_data[replace_indexes_target[i]][634:] = temp[idx][634:]
        if args.client_num == 4:
            for i, idx in enumerate(replace_indexes_others):
                temp[idx][trigger_dimensions] = 1
                new_data[replace_indexes_target[i]][1134:] = temp[idx][1134:]
    else:
        raise Exception("dataset not include")
    logger.info(
        "Add Trigger to %d Bad Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def mode_print(logger, mode):
    if mode == 'train':
        logger.info('=>Add Trigger to Train Data')
    else:
        logger.info('=>Add Trigger to Test Data')
