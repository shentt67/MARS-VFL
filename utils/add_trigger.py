import math
from utils.utils import *


def add_trigger_to_data(args, logger, poison_indexes, new_data, trigger_dimensions, new_targets, rate, mode,
                        replace_label):
    mode_print(logger, mode)
    if args.dataset == 'CIFAR10':
        new_data, new_targets = add_triangle_pattern_trigger(args, logger, poison_indexes, new_data, new_targets, rate,
                                                             mode, replace_label)
        return new_data, new_targets
    elif args.dataset == 'UCIHAR':
        new_data, new_targets = add_feature_trigger(args, logger, poison_indexes, trigger_dimensions, new_data,
                                                    new_targets, rate, mode,
                                                    replace_label)
        return new_data, new_targets
    elif args.dataset == 'PHISHING':
        new_data, new_targets = add_vector_replacement_trigger(args, logger, poison_indexes, trigger_dimensions,
                                                               new_data,
                                                               new_targets, rate, mode,
                                                               replace_label)
        return new_data, new_targets
    elif args.dataset == 'NUSWIDE':
        new_data, new_targets = add_vector_replacement_trigger(args, logger, poison_indexes, trigger_dimensions,
                                                               new_data, new_targets,
                                                               rate,
                                                               mode, replace_label)
        return new_data, new_targets


def add_triangle_pattern_trigger(args, logger, poison_indexes, new_data, new_targets, rate, mode, replace_label):
    height, width, channels = new_data.shape[1:]
    for idx in poison_indexes:
        if replace_label and mode == 'train':
            new_targets[idx] = args.target_label
        for c in range(channels):
            new_data[idx, height - 3:, width - 3:, c] = 0
            new_data[idx, height - 3, width - 1, c] = 255
            new_data[idx, height - 1, width - 3, c] = 255
            new_data[idx, height - 2, width - 2, c] = 255
            new_data[idx, height - 1, width - 1, c] = 255
    logger.info(
        "Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def add_feature_trigger(args, logger, poison_indexes, trigger_dimensions, new_data, new_targets, rate, mode,
                        replace_label=True):
    for idx in poison_indexes:
        new_data[idx][trigger_dimensions] = args.trigger_feature_clip
    logger.info(
        "Add Trigger to %d Bad Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def add_vector_replacement_trigger(args, logger, poison_indexes, trigger_dimensions, new_data, new_targets, rate, mode,
                                   replace_label):
    for idx in poison_indexes:
        if replace_label and mode == 'train':
            new_targets[idx] = args.target_label
        new_data[idx][trigger_dimensions] = 1
    logger.info(
        "Add Trigger to %d Bad Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def mode_print(logger, mode):
    if mode == 'train':
        logger.info('=>Add Trigger to Train Data')
    else:
        logger.info('=>Add Trigger to Test Data')
