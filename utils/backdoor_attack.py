import copy

from utils.add_trigger import add_trigger_to_data
from utils.add_trigger_replace import add_trigger_to_data_replace
from utils.utils import *
from model.MUSTARD_models import*

def attack_lra(args, logger, data, trigger_dimensions, targets, rate, mode):
    new_data = copy.deepcopy(data)
    new_targets = copy.deepcopy(targets)
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]
    new_data, new_targets = add_trigger_to_data(args, logger, poison_indexes, new_data, trigger_dimensions, new_targets,
                                                rate, mode,
                                                replace_label=True)
    return new_data, new_targets


def attack_rsa(args, logger, data, trigger_dimensions, rate, mode):
    new_data = copy.deepcopy(data)
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]
    new_data, _ = add_trigger_to_data(args, logger, poison_indexes, data, trigger_dimensions, [], rate, mode,
                                      replace_label=False)
    return new_data


def attack_sim(args, logger, replace_indexes_others, replace_indexes_target, train_indexes, poison_indexes, data,
               target, trigger_dimensions, rate,
               mode, length = None):
    if args.poison_all:
        new_data, _ = add_trigger_to_data(args, logger, poison_indexes, data, trigger_dimensions, target, rate, mode,
                                          replace_label=False)
    else:
        new_data, _ = add_trigger_to_data_replace(args, logger, replace_indexes_others, replace_indexes_target,
                                                  train_indexes, poison_indexes, data, trigger_dimensions, target, rate,
                                                  mode,
                                                  replace_label=False, length = length)
    return new_data


def select_sim(train_features, num_poisons):
    anchor_idx = get_anchor_sim(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_near_index(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index, anchor_idx


def get_anchor_sim(train_features, num_poisons):
    similarity = train_features @ train_features.T
    w = torch.cat((torch.ones((num_poisons)),
                   -torch.ones((num_poisons))), dim=0)
    top_sim = torch.topk(similarity, 2 * num_poisons, dim=1)[0]
    mean_top_sim = torch.matmul(top_sim, w)
    idx = torch.argmax(mean_top_sim)
    return idx


def get_near_index(anchor_feature, train_features, num_poisons):
    anchor_feature_l1 = torch.norm(anchor_feature, p=1)
    train_features_l1 = torch.norm(train_features, p=1, dim=1)
    vals, indices = torch.topk(torch.div((train_features @ anchor_feature), (train_features_l1 * anchor_feature_l1)),
                               k=num_poisons, dim=0)
    return indices


def get_trigger_dimensions(args, dataset):
    if args.dataset == 'CIFAR10':
        trigger_dimensions = []
        pass
    elif args.dataset == 'UCIHAR':
        if args.client_num == 2:
            # Acceleration Sensor: 1-120,201-239,266-423,503-528,555-556,559-561 (348 in total)
            # Gyroscope Sensor: 121-200, 240-265, 424-502, 529-554, 557-558 (213 in total)
            acc_ind = list(range(0, 120)) + list(range(200, 239)) + list(range(265, 423)) + list(
                range(502, 528)) + list(range(554, 556)) + list(range(558, 561))
            gyr_ind = list(range(120, 200)) + list(range(239, 265)) + list(range(423, 502)) + list(
                range(528, 554)) + list(range(556, 558))
            candidate = [acc_ind, gyr_ind]
            trigger_dimensions = np.random.choice(candidate[args.attack_client], args.poison_dimensions, replace=False)
        elif args.client_num == 4:
            ranges = range(421, 561)
            trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
    elif args.dataset == 'PHISHING':
        if args.client_num == 2:
            ranges = range(15, 30)
            trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
        elif args.client_num == 4:
            ranges = range(23, 30)
            trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
    elif args.dataset == 'NUSWIDE':
        if args.client_num == 2:
            ranges = range(634, 1634)
            trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
        elif args.client_num == 4:
            ranges = range(1134, 1634)
            trigger_dimensions = np.random.choice(ranges, args.poison_dimensions, replace=False)
    else:
        raise_dataset_exception()
    return trigger_dimensions
