import os

import pandas as pd
import numpy as np

import torch

from utils.utils import raise_dataset_exception, raise_split_exception
from itertools import chain, combinations
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
import pickle
import random

from dataset.utils.get_data import PushTask, split_trajectories
import argparse
import fannypack
import copy
from tqdm import tqdm
from .feature_robust import gaussian_noise, misaligned
from dataset.utils import augmentations


class ProcessForce(object):
    """Truncate a time series of force readings with a window size.
    Args:
        window_size (int): Length of the history window that is
            used to truncate the force readings
    """

    def __init__(self, window_size, key='force', tanh=False):
        """Initialize ProcessForce object.

        Args:
            window_size (int): Windows size
            key (str, optional): Key where data is stored. Defaults to 'force'.
            tanh (bool, optional): Whether to apply tanh to output or not. Defaults to False.
        """
        assert isinstance(window_size, int)
        self.window_size = window_size
        self.key = key
        self.tanh = tanh

    def __call__(self, sample):
        """Get data from sample."""
        force = sample[self.key]
        force = force[-self.window_size:]
        if self.tanh:
            force = np.tanh(force)  # remove very large force readings
        sample[self.key] = force.transpose()
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        """Initialize ToTensor object."""
        self.device = device

    def __call__(self, sample):
        """Convert sample argument from ndarray with H,W,C dimensions to a tensor with C,H,W dimensions."""
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # transpose flow into 2 x H x W
        for k in sample.keys():
            if k.startswith('flow'):
                sample[k] = sample[k].transpose((2, 0, 1))

        # convert numpy arrays to pytorch tensors
        new_dict = dict()
        for k, v in sample.items():
            if self.device is None:
                # torch.tensor(v, device = self.device, dtype = torch.float32)
                new_dict[k] = torch.FloatTensor(v)
            else:
                new_dict[k] = torch.from_numpy(v).float()

        return new_dict


def aug(x):
    mixture_width = 3
    mixture_depth = -1
    aug_severity = 3
    all_ops = True

    aug_list = augmentations.augmentations_general

    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = np.zeros_like(x)
    for i in range(mixture_width):
        x_aug = copy.deepcopy(x)
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            x_aug = op(x_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += np.array(ws[i] * x_aug)

    mixed = (1 - m) * x + m * mix
    return mixed


def augment_val(val_filename_list, filename_list):
    """Augment lists of filenames so that they match the current directory."""
    filename_list1 = copy.deepcopy(filename_list)
    val_filename_list1 = []

    for name in tqdm(val_filename_list):
        filename = name[:-8]
        found = True

        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        if file_number < 10:
            comp_number = 19
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                    filename1 not in val_filename_list1
            ):
                comp_number += -1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number < 0:
                    found = False
                    break
        else:
            comp_number = 0
            filename1 = filename + str(comp_number) + "_1000.h5"
            while (filename1 not in filename_list1) and (
                    filename1 not in val_filename_list1
            ):
                comp_number += 1
                filename1 = filename + str(comp_number) + "_1000.h5"
                if comp_number > 19:
                    found = False
                    break

        if found:
            if filename1 in filename_list1:
                filename_list1.remove(filename1)

            if filename1 not in val_filename_list:
                val_filename_list1.append(filename1)

    val_filename_list1 += val_filename_list

    return val_filename_list1, filename_list1

from typing import Iterator, Optional, Sequence, TypeVar, Generic, Sized

T_co = TypeVar('T_co', covariant=True)


class Sampler(Generic[T_co]):
    r"""Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

class SubsetRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int], generator=None) -> None:
        self.indices = indices
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        for i in torch.randperm(len(self.indices), generator=self.generator):
            yield self.indices[i]

    def __len__(self) -> int:
        return len(self.indices)

def combine_modalitiesbuilder(unimodal, output):
    """Create a function data combines modalities given the type of input.

    Args:
        unimodal (str): Input type as a string. Can be 'force', 'proprio', 'image'. Defaults to using all modalities otherwise
        output (int): Index of output modality.
    """

    def _combine_modalities(data):
        if unimodal == "force":
            return [data['force'], data['action'], data[output]]
        if unimodal == "proprio":
            return [data['proprio'], data['action'], data[output]]
        if unimodal == "image":
            return [data['image'], data['depth'].transpose(0, 2).transpose(1, 2), data['action'], data[output]]
        return [
            data['image'],
            data['force'],
            data['proprio'],
            data['depth'].transpose(0, 2).transpose(1, 2),
            data['action'],
            data[output],
        ]

    return _combine_modalities


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        print("Loading {}.".format(file))
        df = pd.read_csv(file, header=None, engine="c")
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    # get XA, which are image low level features
    features_path = "Low_Level_Features"
    dfs = []
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            print("Loading {}.".format(os.path.join(data_dir, features_path, file)))
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ", engine="c")
            df.dropna(axis=1, inplace=True)
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]
    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    print("Loading {}.".format(file))
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t", engine="c")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]
    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], np.argmax(selected.values[:], 1)
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], np.argmax(
        selected.values[:n_samples])


def drop_entry(dataset):
    """Drop entries where there's no text in the data."""
    drop = []
    for ind, k in enumerate(dataset["text"]):
        if k.sum() == 0:
            drop.append(ind)

    for modality in list(dataset.keys()):
        dataset[modality] = np.delete(dataset[modality], drop, 0)
    return dataset


def get_vfl_data_distribution(args):
    if args.dataset == 'NUSWIDE':
        if args.client_num == 2:
            # Image Features: 0-634 (634 in total)
            # Text Features: 634-1634 (1000 in total)
            return [list(range(0, 634)), list(range(634, 1634))]
        else:
            raise_split_exception()
    if args.dataset == 'UCIHAR':
        if args.client_num == 2:
            # Acceleration Sensor: 1-120,201-239,266-423,503-528,555-556,559-561 (348 in total)
            # Gyroscope Sensor: 121-200, 240-265, 424-502, 529-554, 557-558 (213 in total)
            acc_ind = list(range(0, 120)) + list(range(200, 239)) + list(range(265, 423)) + list(
                range(502, 528)) + list(range(554, 556)) + list(range(558, 561))
            gyr_ind = list(range(120, 200)) + list(range(239, 265)) + list(range(423, 502)) + list(
                range(528, 554)) + list(range(556, 558))
            return [acc_ind, gyr_ind]
        else:
            raise_split_exception()
    if args.dataset == 'KUHAR':
        if args.client_num == 2:
            # Acceleration Sensor: 1-900
            # Gyroscope Sensor: 901-1800
            acc_ind = list(range(0, 900))
            gyr_ind = list(range(900, 1800))
            return [acc_ind, gyr_ind]
        else:
            raise_split_exception()
    if args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
        if args.client_num == 3:
            return ['vision', 'audio', 'text']
        else:
            raise_split_exception()
    if args.dataset in ['MIMIC', 'MMIMDB']:
        if args.client_num == 2:
            return [0, 1]
        else:
            raise_split_exception()
    if args.dataset in ['MUJOCO']:
        if args.client_num == 4:
            return [0, 1, 2, 3]
        else:
            raise_split_exception()
    if args.dataset in ['VISIONTOUCH']:
        if args.client_num == 5:
            return [0, 1, 2, 3, 4]
        else:
            raise_split_exception()
    if args.dataset in ['PTBXL']:
        return [0, 1]


def prepare_data_vfl(data, args):
    if args.dataset in ['NUSWIDE', 'UCIHAR', 'KUHAR']:
        if args.client_num == 2:
            distribution = get_vfl_data_distribution(args)
            x_a = data[:, distribution[0]]
            x_b = data[:, distribution[1]]
            return [x_a, x_b]
        else:
            raise_split_exception()
    if args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
        if args.client_num == 3:
            return data
        else:
            raise_split_exception()
    if args.dataset in ['MIMIC', 'MMIMDB']:
        if args.client_num == 2:
            return data
        else:
            raise_split_exception()
    if args.dataset in ['PTBXL']:
        if args.client_num == 3:
            return [data[0], data[1][:, :, 0:6], data[1][:, :, 6:12]]
    if args.dataset in ['MUJOCO']:
        if args.client_num == 4:
            return data
        else:
            raise_split_exception()
    if args.dataset in ['VISIONTOUCH']:
        if args.client_num == 5:
            return data
        else:
            raise_split_exception()
    else:
        raise_dataset_exception()


def _process_emo(inputs, args):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []
    mask_list = []

    for i in range(len(inputs[0]) - 3):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        pad_seq = torch.where(torch.isinf(pad_seq), 0, pad_seq)
        processed_input.append(pad_seq)

    for sample in inputs:

        inds.append(sample[-3])

        mask_list.append(sample[-1])

        if args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
            labels.append(sample[-2])

    return processed_input, processed_input_lengths, torch.tensor(inds).view(len(inputs), 1), \
           torch.tensor(labels).view(len(inputs), 1), mask_list[0]


def _process_emo_robustness(inputs, args):
    processed_input = []
    processed_input_lengths = []
    inds = []
    labels = []
    mask_list = []

    for i in range(len(inputs[0]) - 3):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.size(0) for v in feature]))
        pad_seq = pad_sequence(feature, batch_first=True)
        pad_seq = torch.where(torch.isinf(pad_seq), 0, pad_seq)
        processed_input.append(pad_seq)

    for sample in inputs:

        inds.append(sample[-2])

        mask_list.append(sample[-1])

        if args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
            labels.append(sample[-2])

    return processed_input[0:3], processed_input[3:6], processed_input[6:9], processed_input_lengths, torch.tensor(
        inds).view(len(inputs), 1), \
           torch.tensor(labels).view(len(inputs), 1), mask_list[0]


def _get_mask_per_batch(num_batches, num_blocks, p_miss=None):
    p_observed = None if p_miss is None else (1 - p_miss)
    if isinstance(p_observed, float) or isinstance(p_observed, int):
        p_observed_array = np.array([p_observed] * num_blocks)
    elif p_observed is None:
        p_observed_array = np.random.beta(2.0, 2.0, num_blocks)

    patterns = [np.array([bool(int(x)) for x in bin(i)[2:].zfill(num_blocks)]) for i in range(2 ** num_blocks)]

    probabilities = []
    for pattern in patterns:
        prob = 1.0
        for block, p_observed in zip(pattern, p_observed_array):
            prob *= p_observed if block else (1 - p_observed)
        probabilities.append(prob)

    chosen_patterns = np.random.choice(len(patterns), size=num_batches,
                                       p=np.array(probabilities) / np.sum(probabilities))
    batch_patterns = [patterns[i] for i in chosen_patterns]

    return batch_patterns, p_observed_array


def powerset_except_empty(n):
    return list(chain.from_iterable(combinations(range(n), r) for r in range(1, n + 1)))


def _process_none(inputs, args):
    processed_input = []
    processed_input_lengths = []
    mask_list = []
    inds = []
    labels = []
    for i in range(len(inputs[0]) - 4):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.shape[0] for v in feature]))
        if isinstance(feature[0], torch.Tensor):
            feature = torch.stack(feature)
        elif isinstance(feature[0], np.ndarray):
            feature = torch.tensor(feature)
        processed_input.append(feature)
    for sample in inputs:
        inds.append(sample[-3])
        labels.append(sample[-2])
        mask_list.append(sample[-1])

    if args.dataset in ['PTBXL', 'MIMIC']:
        return processed_input, processed_input_lengths, torch.tensor(inds).view(len(inputs), 1), \
               torch.stack(labels), mask_list[0]

    if args.dataset in ['MMIMDB']:
        return processed_input, processed_input_lengths, torch.tensor(inds).view(len(inputs), 1), \
                               torch.tensor(labels), mask_list[0]

    if args.dataset in ['VISIONTOUCH']:
        return processed_input, processed_input_lengths, torch.tensor(inds).view(len(inputs), 1), \
            torch.tensor(labels).view(len(inputs), 1), mask_list[0]

    if args.dataset in ['VISIONTOUCH', 'UCIHAR', 'KUHAR', 'NUSWIDE']:
        return processed_input[0], processed_input_lengths, torch.tensor(inds).view(len(inputs), 1), \
               torch.tensor(labels).view(len(inputs), 1), mask_list[0]

    if args.dataset in ['MUJOCO']:
        return processed_input, processed_input_lengths, torch.tensor(inds).view(len(inputs), 1), \
                       torch.stack(labels), mask_list[0]


def _process_none_robustness(inputs, args):
    processed_input = []
    processed_input_lengths = []
    mask_list = []
    inds = []
    labels = []
    for i in range(len(inputs[0]) - 4):
        feature = []
        for sample in inputs:
            feature.append(sample[i])
        processed_input_lengths.append(torch.as_tensor([v.shape[0] for v in feature]))
        if isinstance(feature[0], torch.Tensor):
            feature = torch.stack(feature)
        elif isinstance(feature[0], np.ndarray):
            feature = torch.tensor(feature)
        processed_input.append(feature)
    for sample in inputs:
        inds.append(sample[-3])
        labels.append(sample[-2])
        mask_list.append(sample[-1])

    if args.dataset in ['MUJOCO']:
        return processed_input[0:4], processed_input[4:8], processed_input[8:12], processed_input_lengths, torch.tensor(
            inds).view(len(inputs), 1), \
               torch.stack(labels), mask_list[0]



    if args.dataset in ['PTBXL', 'MIMIC']:
        return processed_input[0:2], processed_input[2:4], processed_input[
                                                           4:6], processed_input_lengths, torch.tensor(inds).view(
            len(inputs), 1), \
               torch.stack(labels), mask_list[0]

    if args.dataset in ['MMIMDB']:
        return processed_input[0:2], processed_input[2:4], processed_input[
                                                           4:6], processed_input_lengths, torch.tensor(inds).view(
            len(inputs), 1), \
            torch.tensor(labels), mask_list[0]

    if args.dataset in ['VISIONTOUCH']:
        return processed_input[0:5], processed_input[5:10], processed_input[
                                                            10:15], processed_input_lengths, torch.tensor(
            inds).view(len(inputs), 1), \
            torch.stack(labels), mask_list[0]

    if args.dataset in ['UCIHAR', 'KUHAR', 'NUSWIDE']:
        return processed_input[0], processed_input[1], processed_input[2], processed_input_lengths, torch.tensor(
            inds).view(len(inputs), 1), \
               torch.tensor(labels).view(len(inputs), 1), mask_list[0]


class Human_MM_VFL(Dataset):
    def __init__(self, args, dtype):
        self.args = args
        if args.dataset == 'UCIHAR':
            train_data = np.loadtxt(args.data_dir + 'UCIHAR/UCI HAR Dataset/train/X_train.txt')
            train_targets = np.loadtxt(args.data_dir + 'UCIHAR/UCI HAR Dataset/train/y_train.txt') - 1
            test_data = np.loadtxt(args.data_dir + 'UCIHAR/UCI HAR Dataset/test/X_test.txt')
            test_targets = np.loadtxt(args.data_dir + 'UCIHAR/UCI HAR Dataset/test/y_test.txt') - 1
        elif args.dataset == 'KUHAR':
            data = np.loadtxt(open(args.data_dir + 'KUHAR/KU-HAR_20750x300.csv', 'rb'), delimiter=",")
            X = data[:, 0:1800]
            y = data[:, 1800]
            y = y.reshape((len(y), 1))
            X = torch.tensor(X)
            y = torch.tensor(y)
            indexes_list = np.array(range(len(X)))
            random.seed(args.seed)
            train_indexes = random.sample(list(indexes_list), int(20750 * 0.8))
            test_indexes = list(set(list(indexes_list)).difference(set(train_indexes)))
            train_data, test_data = X[train_indexes], X[test_indexes]
            train_targets, test_targets = y[train_indexes].reshape(-1), y[test_indexes].reshape(-1)
        elif args.dataset == 'NUSWIDE':
            selected_labels = ['buildings', 'grass', 'animal', 'water', 'person']
            X_image_train, X_text_train, Y_train = get_labeled_data(args.data_dir + 'NUS_WIDE', selected_labels, None,
                                                                    'Train')
            train_data = torch.cat((torch.tensor(X_image_train), torch.tensor(X_text_train)), dim=1)
            train_targets = torch.tensor(Y_train)
            X_image_test, X_text_test, Y_test = get_labeled_data(args.data_dir + 'NUS_WIDE', selected_labels, None,
                                                                 'Test')
            test_data = torch.cat((torch.tensor(X_image_test), torch.tensor(X_text_test)), dim=1)
            test_targets = torch.tensor(Y_test)

        if dtype == 'train':
            self.data = train_data
            self.data_p = copy.deepcopy(self.data)
            self.targets = train_targets
            self.missing_rate = args.p_miss_train
        elif dtype == 'valid':
            self.data = test_data
            self.data_p = copy.deepcopy(self.data)
            self.targets = test_targets
            self.missing_rate = args.p_miss_test

        num_samples = len(self.data)
        num_batches = (num_samples + args.batch_size - 1) // args.batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(num_batches, args.client_num, self.missing_rate)
        self.batch_size = args.batch_size

        if args.eval_mode == 'robustness':
            import collections
            d = collections.OrderedDict()
            d['Gaussian Noise'] = gaussian_noise
            # d['Random Perturbation'] = random_perturb
            idxes = np.random.permutation(len(self.data))
            if dtype == 'train':
                perturb_data_len = int(args.perturb_rate_train * len(self.data))
            elif dtype == 'valid':
                perturb_data_len = int(args.perturb_rate_test * len(self.data))
            idxes = idxes[0:perturb_data_len]
            distribution = get_vfl_data_distribution(args)
            if args.perturb_type == 'missing':
                pass
            elif args.perturb_type == 'corrupted':
                for index in idxes:
                    method_name = random.sample(d.keys(), 1)[0]
                    severity = np.random.randint(1, 5)
                    corruption = lambda clean_data: d[method_name](clean_data, severity)
                    # different data part with different corruptions
                    for i in range(len(distribution)):
                        self.data[index][distribution[i]] = corruption(self.data[index][distribution[i]])
            elif args.perturb_type == 'misaligned':
                list_misaligned_clients = list(range(0, self.args.client_num))
                list_misaligned_clients.remove(args.anchor_client)
                list_misaligned_data = []
                for client in iter(list_misaligned_clients):
                    misaligned_data = misaligned(self.data[idxes][:, distribution[client]])
                    list_misaligned_data.append(misaligned_data)
                for i in range(len(idxes)):
                    for client in iter(list_misaligned_clients):
                        if client < args.anchor_client:
                            self.data[idxes[i]][distribution[client]] = list_misaligned_data[client][i]
                        else:
                            self.data[idxes[i]][distribution[client]] = list_misaligned_data[client - 1][i]

            self.data_aug_1 = copy.deepcopy(self.data)
            self.data_aug_2 = copy.deepcopy(self.data)
            for i in range(len(self.data)):
                for j in range(len(distribution)):
                    self.data_aug_1[i][distribution[j]] = aug(self.data_aug_1[i][distribution[j]])
                    self.data_aug_2[i][distribution[j]] = aug(self.data_aug_2[i][distribution[j]])

    def __getitem__(self, index):
        batch_idx = index // self.batch_size

        x = self.data[index]
        length = []
        y = self.targets[index]
        mask = self.batch_patterns[batch_idx]

        if self.args.eval_mode == 'robustness':
            x_aug_1 = self.data_aug_1[index]
            x_aug_2 = self.data_aug_2[index]
            return [x, x_aug_1, x_aug_2, length, index, y, torch.tensor(mask, dtype=torch.bool)]

        return [x, length, index, y, torch.tensor(mask, dtype=torch.bool)]

    def __len__(self):
        return len(self.data)


class Emotion_VFL(Dataset):
    def __init__(self, args, dtype):
        self.args = args
        if args.dataset == 'MOSI':
            with open(args.data_dir + "MOSI/mosi_raw.pkl", "rb") as f:
                alldata = pickle.load(f)
        elif args.dataset == 'MOSEI':
            with open(args.data_dir + "MOSEI/mosei_senti_data.pkl", "rb") as f:
                alldata = pickle.load(f)
        elif args.dataset == 'URFUNNY':
            with open(args.data_dir + "UR-FUNNY/humor.pkl", "rb") as f:
                alldata = pickle.load(f)
        elif args.dataset == 'MUSTARD':
            with open(args.data_dir + "MUSTARD/sarcasm.pkl", "rb") as f:
                alldata = pickle.load(f)
        if dtype == 'train':
            alldata['train'] = drop_entry(alldata['train'])
            self.data = alldata['train']
            self.data_p = copy.deepcopy(self.data)
            self.targets = torch.tensor(alldata['train']['labels'])
            self.missing_rate = args.p_miss_train
        elif dtype == 'valid':
            alldata['valid'] = drop_entry(alldata['valid'])
            self.data = alldata['valid']
            self.data_p = copy.deepcopy(self.data)
            self.targets = torch.tensor(alldata['valid']['labels'])
            self.missing_rate = args.p_miss_train
        elif dtype == 'test':
            alldata['test'] = drop_entry(alldata['test'])
            self.data = alldata['test']
            self.data_p = copy.deepcopy(self.data)
            self.targets = torch.tensor(alldata['test']['labels'])
            self.missing_rate = args.p_miss_test

        num_samples = len(self.data['vision'])
        num_batches = (num_samples + args.batch_size - 1) // args.batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(num_batches, args.client_num, self.missing_rate)
        self.batch_size = args.batch_size

        if args.eval_mode == 'robustness':
            import collections
            d = collections.OrderedDict()
            d['Gaussian Noise'] = gaussian_noise
            idxes = np.random.permutation(num_samples)
            if dtype == 'train':
                perturb_data_len = int(args.perturb_rate_train * num_samples)
            elif dtype == 'valid':
                perturb_data_len = int(args.perturb_rate_train * num_samples)
            elif dtype == 'test':
                perturb_data_len = int(args.perturb_rate_test * num_samples)
            idxes = idxes[0:perturb_data_len]
            distribution = get_vfl_data_distribution(args)
            if args.perturb_type == 'missing':
                pass
            elif args.perturb_type == 'corrupted':
                for index in idxes:
                    method_name = random.sample(d.keys(), 1)[0]
                    severity = np.random.randint(1, 5)
                    corruption = lambda clean_data: d[method_name](clean_data, severity)
                    # different data part with different corruptions
                    for i in range(len(distribution)):
                        self.data[distribution[i]][index] = corruption(self.data[distribution[i]][index])
            elif args.perturb_type == 'misaligned':
                list_misaligned_clients = list(range(0, self.args.client_num))
                list_misaligned_clients.remove(args.anchor_client)
                list_misaligned_data = []
                for client in iter(list_misaligned_clients):
                    misaligned_data = misaligned(self.data[distribution[client]][idxes])
                    list_misaligned_data.append(misaligned_data)
                for i in range(len(idxes)):
                    for client in iter(list_misaligned_clients):
                        if client < args.anchor_client:
                            self.data[distribution[client]][idxes[i]] = list_misaligned_data[client][i]
                        else:
                            self.data[distribution[client]][idxes[i]] = list_misaligned_data[client - 1][i]
            self.data_aug_1 = copy.deepcopy(self.data)
            self.data_aug_2 = copy.deepcopy(self.data)
            for i in range(len(self.data['vision'])):
                for j in range(len(distribution)):
                    self.data_aug_1[distribution[j]][i] = aug(self.data_aug_1[distribution[j]][i])
                    self.data_aug_2[distribution[j]][i] = aug(self.data_aug_2[distribution[j]][i])

    def __getitem__(self, index):
        batch_idx = index // self.batch_size
        vision = torch.tensor(self.data['vision'][index])
        audio = torch.tensor(self.data['audio'][index])
        text = torch.tensor(self.data['text'][index])

        mask = self.batch_patterns[batch_idx]
        try:
            start = text.nonzero(as_tuple=False)[0][0]
        except:
            print(text, index)
            exit()
        vision = vision[start:].float()
        audio = audio[start:].float()
        text = text[start:].float()
        tmp_label = self.data['labels'][index]
        label = torch.tensor(tmp_label).float()
        if self.args.eval_mode == 'robustness':
            vision_aug_1 = torch.tensor(self.data_aug_1['vision'][index])
            audio_aug_1 = torch.tensor(self.data_aug_1['audio'][index])
            text_aug_1 = torch.tensor(self.data_aug_1['text'][index])

            vision_aug_2 = torch.tensor(self.data_aug_2['vision'][index])
            audio_aug_2 = torch.tensor(self.data_aug_2['audio'][index])
            text_aug_2 = torch.tensor(self.data_aug_2['text'][index])

            vision_aug_1 = vision_aug_1[start:].float()
            audio_aug_1 = audio_aug_1[start:].float()
            text_aug_1 = text_aug_1[start:].float()

            vision_aug_2 = vision_aug_2[start:].float()
            audio_aug_2 = audio_aug_2[start:].float()
            text_aug_2 = text_aug_2[start:].float()
            return [vision, audio, text, vision_aug_1, audio_aug_1, text_aug_1, vision_aug_2, audio_aug_2, text_aug_2,
                    index, label, torch.tensor(mask, dtype=torch.bool)]
        return [vision, audio, text, index, label, torch.tensor(mask, dtype=torch.bool)]

    def __len__(self):
        return self.data['vision'].shape[0]


class Healthcare_VFL(Dataset):
    def __init__(self, args, dtype, dataset):
        self.args = args
        if dataset == 'MIMIC_VFL':
            f = open(args.data_dir + "MIMIC/im.pk", 'rb')
            datafile = pickle.load(f)
            f.close()
            X_t = datafile['ep_tdata']
            X_s = datafile['adm_features_all']

            X_t[np.isinf(X_t)] = 0
            X_t[np.isnan(X_t)] = 0
            X_s[np.isinf(X_s)] = 0
            X_s[np.isnan(X_s)] = 0

            X_s_avg = np.average(X_s, axis=0)
            X_s_std = np.std(X_s, axis=0)
            X_t_avg = np.average(X_t, axis=(0, 1))
            X_t_std = np.std(X_t, axis=(0, 1))

            for i in range(len(X_s)):
                X_s[i] = (X_s[i] - X_s_avg) / X_s_std
                for j in range(len(X_t[0])):
                    X_t[i][j] = (X_t[i][j] - X_t_avg) / X_t_std

            static_dim = len(X_s[0])
            timestep = len(X_t[0])
            series_dim = len(X_t[0][0])
            y = datafile['y_icd9'][:, 7]
            le = len(y)

            datasets = [(X_s[i], X_t[i], y[i]) for i in range(le)]

            indexes_list = np.array(range(len(datasets)))
            random.seed(args.seed)
            train_indexes = random.sample(list(indexes_list), int(len(datasets) * 0.8))
            valid_test_indexes = list(set(list(indexes_list)).difference(set(train_indexes)))
            valid_indexes = random.sample(list(indexes_list), int(len(datasets) * 0.5))
            test_indexes = list(set(list(valid_test_indexes)).difference(set(valid_indexes)))
            train_data, valid_data, test_data = [datasets[i] for i in train_indexes], [datasets[i] for i in
                                                                                       valid_indexes], [datasets[i] for
                                                                                                        i in
                                                                                                        test_indexes]

            if dtype == 'train':
                self.data = train_data
                self.missing_rate = args.p_miss_train
            elif dtype == 'valid':
                self.data = valid_data
                self.missing_rate = args.p_miss_train
            elif dtype == 'test':
                self.data = test_data
                self.missing_rate = args.p_miss_test

        elif dataset == 'PTBXL_VFL':
            with open(args.data_dir + 'PTB-XL/PTBXL.pk', 'rb') as f:
                dataset = pickle.load(f)
                f.close()

            if dtype == 'train':
                data_t = dataset['train']
                self.missing_rate = args.p_miss_train
            elif dtype == 'valid':
                data_t = dataset['valid']
                self.missing_rate = args.p_miss_train
            elif dtype == 'test':
                data_t = dataset['test']
                self.missing_rate = args.p_miss_test
            x_attributes = data_t['attributes'].astype(float)
            x_series = data_t['series']
            targets = data_t['labels']
            x_attributes[np.isinf(x_attributes)] = 0
            x_attributes[np.isnan(x_attributes)] = 0
            x_series[np.isinf(x_series)] = 0
            x_series[np.isnan(x_series)] = 0

            x_attributes_mean = np.mean(x_attributes, axis=0)
            x_attributes_std = np.std(x_attributes, axis=0)
            x_attributes = (x_attributes - x_attributes_mean) / x_attributes_std

            self.data = [[x_attributes[i], x_series[i], targets[i]] for i in range(len(targets))]

        num_samples = len(self.data)
        num_batches = (num_samples + args.batch_size - 1) // args.batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(num_batches, args.client_num, self.missing_rate)
        self.batch_size = args.batch_size

        if args.eval_mode == 'robustness':
            import collections
            d = collections.OrderedDict()
            d['Gaussian Noise'] = gaussian_noise
            idxes = np.random.permutation(len(self.data))
            if dtype == 'train':
                perturb_data_len = int(args.perturb_rate_train * len(self.data))
            elif dtype == 'valid':
                perturb_data_len = int(args.perturb_rate_train * len(self.data))
            elif dtype == 'test':
                perturb_data_len = int(args.perturb_rate_test * len(self.data))
            idxes = idxes[0:perturb_data_len]
            distribution = get_vfl_data_distribution(args)
            if args.perturb_type == 'missing':
                pass
            elif args.perturb_type == 'corrupted':
                for index in idxes:
                    method_name = random.sample(d.keys(), 1)[0]
                    severity = np.random.randint(1, 5)
                    corruption = lambda clean_data: d[method_name](clean_data, severity)
                    # different data part with different corruptions
                    for i in range(len(distribution)):
                        temp_list = list(self.data[idxes[i]])
                        temp_list[distribution[i]] = corruption(temp_list[distribution[i]])
                        self.data[idxes[i]] = tuple(temp_list)
            elif args.perturb_type == 'misaligned':

                list_misaligned_clients = list(range(0, self.args.client_num))
                list_misaligned_clients.remove(args.anchor_client)
                list_misaligned_data = []
                for client in iter(list_misaligned_clients):
                    misaligned_data = misaligned(
                        np.array([self.data[i][distribution[client]] for i in idxes]))
                    list_misaligned_data.append(misaligned_data)
                for i in range(len(idxes)):
                    for client in iter(list_misaligned_clients):
                        if client < args.anchor_client:
                            temp_list = list(self.data[idxes[i]])
                            temp_list[distribution[client]] = list_misaligned_data[client][i]
                            self.data[idxes[i]] = tuple(temp_list)
                        else:
                            temp_list = list(self.data[idxes[i]])
                            temp_list[distribution[client]] = list_misaligned_data[client-1][i]
                            self.data[idxes[i]] = tuple(temp_list)
            self.data_aug_1 = copy.deepcopy(self.data)
            self.data_aug_2 = copy.deepcopy(self.data)
            for i in range(len(self.data)):
                for j in range(len(distribution)):
                    temp_list_aug_1 = list(self.data_aug_1[i])
                    temp_list_aug_1[distribution[j]] = aug(temp_list_aug_1[distribution[j]])
                    self.data_aug_1[i] = tuple(temp_list_aug_1)

                    temp_list_aug_2 = list(self.data_aug_2[i])
                    temp_list_aug_2[distribution[j]] = aug(temp_list_aug_2[distribution[j]])
                    self.data_aug_2[i] = tuple(temp_list_aug_2)

    def __getitem__(self, index):
        batch_idx = index // self.batch_size
        tab_att = self.data[index][0]
        series_att = self.data[index][1]

        mask = self.batch_patterns[batch_idx]
        tmp_label = self.data[index][2]
        label = torch.tensor(tmp_label).float()
        length = []
        if self.args.eval_mode == 'robustness':
            tab_att_aug_1 = self.data_aug_1[index][0]
            series_att_aug_1 = self.data_aug_1[index][1]

            tab_att_aug_2 = self.data_aug_2[index][0]
            series_att_aug_2 = self.data_aug_2[index][1]
            return [tab_att, series_att, tab_att_aug_1, series_att_aug_1, tab_att_aug_2, series_att_aug_2, length,
                    index, label, torch.tensor(mask, dtype=torch.bool)]
        return [tab_att, series_att, length, index, label, torch.tensor(mask, dtype=torch.bool)]

    def __len__(self):
        return len(self.data)


class Robotic_VFL(Dataset):
    def __init__(self, args, dtype):
        self.args = args
        Task = PushTask

        parser = argparse.ArgumentParser()
        Task.add_dataset_arguments(parser)
        args = parser.parse_args()
        dataset_args = Task.get_dataset_args(args)

        fannypack.data.set_cache_path('/data/VFLBench/gentle_push/cache')

        train_trajectories = Task.get_train_trajectories(**dataset_args)
        val_trajectories = Task.get_eval_trajectories(**dataset_args)
        test_trajectories = Task.get_test_trajectories(
            modalities=None, **dataset_args)

        train_subsequence = split_trajectories(
            train_trajectories, subsequence_length=16, modalities=None
        )
        valid_subsequence = split_trajectories(
            val_trajectories, subsequence_length=16, modalities=None
        )
        test_subsequence = split_trajectories(
            test_trajectories, subsequence_length=16, modalities=None
        )

        if dtype == 'train':
            self.data = train_subsequence
            self.missing_rate = args.p_miss_train
        elif dtype == 'valid':
            self.data = valid_subsequence
            self.missing_rate = args.p_miss_train
        elif dtype == 'test':
            self.data = test_subsequence
            self.missing_rate = args.p_miss_test

        num_samples = len(self.data)
        num_batches = (num_samples + self.args.batch_size - 1) // self.args.batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(num_batches, self.args.client_num,
                                                                         self.missing_rate)
        self.batch_size = self.args.batch_size

        if self.args.eval_mode == 'robustness':
            import collections
            d = collections.OrderedDict()
            d['Gaussian Noise'] = gaussian_noise
            idxes = np.random.permutation(len(self.data))
            if dtype == 'train':
                perturb_data_len = int(args.perturb_rate_train * len(self.data))
            elif dtype == 'valid':
                perturb_data_len = int(args.perturb_rate_train * len(self.data))
            elif dtype == 'test':
                perturb_data_len = int(args.perturb_rate_test * len(self.data))
            idxes = idxes[0:perturb_data_len]
            distribution = get_vfl_data_distribution(self.args)
            if args.perturb_type == 'missing':
                pass
            elif args.perturb_type == 'corrupted':
                for index in idxes:
                    method_name = random.sample(d.keys(), 1)[0]
                    severity = np.random.randint(1, 5)
                    corruption = lambda clean_data: d[method_name](clean_data, severity)
                    # different data part with different corruptions
                    for i in range(len(distribution)):
                        temp_list = list(self.data[index])
                        temp_list[distribution[i]] = corruption(temp_list[distribution[i]])
                        self.data[index] = tuple(temp_list)
            elif self.args.perturb_type == 'misaligned':
                list_misaligned_clients = list(range(0, self.args.client_num))
                list_misaligned_clients.remove(args.anchor_client)
                list_misaligned_data = []
                for client in iter(list_misaligned_clients):
                    misaligned_data = misaligned(
                        np.array([self.data[i][distribution[client]] for i in idxes]))
                    list_misaligned_data.append(misaligned_data)
                for i in range(len(idxes)):
                    for client in iter(list_misaligned_clients):
                        if client < args.anchor_client:
                            temp_list = list(self.data[idxes[i]])
                            temp_list[distribution[client]] = list_misaligned_data[client][i]
                            self.data[idxes[i]] = tuple(temp_list)
                        else:
                            temp_list = list(self.data[idxes[i]])
                            temp_list[distribution[client]] = list_misaligned_data[client-1][i]
                            self.data[idxes[i]] = tuple(temp_list)
            self.data_aug_1 = copy.deepcopy(self.data)
            self.data_aug_2 = copy.deepcopy(self.data)
            for i in range(len(self.data)):
                for j in range(len(distribution)):
                    temp_list_aug_1 = list(self.data_aug_1[i])
                    temp_list_aug_1[distribution[j]] = aug(temp_list_aug_1[distribution[j]])
                    self.data_aug_1[i] = tuple(temp_list_aug_1)

                    temp_list_aug_2 = list(self.data_aug_2[i])
                    temp_list_aug_2[distribution[j]] = aug(temp_list_aug_2[distribution[j]])
                    self.data_aug_2[i] = tuple(temp_list_aug_2)

    def __getitem__(self, index):
        batch_idx = index // self.batch_size
        pos = self.data[index][0]
        sensor = self.data[index][1]
        image = self.data[index][2]
        control = self.data[index][3]

        mask = self.batch_patterns[batch_idx]
        tmp_label = self.data[index][4]
        label = torch.tensor(tmp_label).float()
        length = []
        if self.args.eval_mode == 'robustness':
            pos_aug_1 = self.data_aug_1[index][0]
            sensor_aug_1 = self.data_aug_1[index][1]
            image_aug_1 = self.data_aug_1[index][2]
            control_aug_1 = self.data_aug_1[index][3]

            pos_aug_2 = self.data_aug_2[index][0]
            sensor_aug_2 = self.data_aug_2[index][1]
            image_aug_2 = self.data_aug_2[index][2]
            control_aug_2 = self.data_aug_2[index][3]
            return [pos, sensor, image, control, pos_aug_1, sensor_aug_1, image_aug_1, control_aug_1, pos_aug_2,
                    sensor_aug_2, image_aug_2, control_aug_2, length, index, label,
                    torch.tensor(mask, dtype=torch.bool)]
        return [pos, sensor, image, control, length, index, label, torch.tensor(mask, dtype=torch.bool)]

    def __len__(self):
        return len(self.data)
