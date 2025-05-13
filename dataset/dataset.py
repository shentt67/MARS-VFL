from torch.utils.data import Dataset
from .utils.utils import augment_val, Human_MM_VFL, Emotion_VFL, Healthcare_VFL, Robotic_VFL, \
    combine_modalitiesbuilder, _get_mask_per_batch
import torch
import yaml
import os
import numpy as np
import h5py
from tqdm import tqdm
import torchvision
from dataset.utils.utils import ProcessForce
from dataset.utils.utils import ToTensor
from dataset.utils.feature_robust import gaussian_noise, misaligned
from dataset.utils.utils import get_vfl_data_distribution, aug
import random
import copy


class UCIHAR_VFL(Human_MM_VFL):
    def __init__(self, args, dtype):
        Human_MM_VFL.__init__(self, args, dtype)


class KUHAR_VFL(Human_MM_VFL):
    def __init__(self, args, dtype):
        Human_MM_VFL.__init__(self, args, dtype)


class NUSWIDE_VFL(Human_MM_VFL):
    def __init__(self, args, dtype):
        Human_MM_VFL.__init__(self, args, dtype)


class MOSI_VFL(Emotion_VFL):
    def __init__(self, args, dtype):
        Emotion_VFL.__init__(self, args, dtype)


class MOSEI_VFL(Emotion_VFL):
    def __init__(self, args, dtype):
        Emotion_VFL.__init__(self, args, dtype)


class URFUNNY_VFL(Emotion_VFL):
    def __init__(self, args, dtype):
        Emotion_VFL.__init__(self, args, dtype)


class MUSTARD_VFL(Emotion_VFL):
    def __init__(self, args, dtype):
        Emotion_VFL.__init__(self, args, dtype)


class MIMIC_VFL(Healthcare_VFL):
    def __init__(self, args, dtype):
        # Healthcare_VFL.__init__(self, root + "MIMIC/im.pk", dtype, transforms)
        Healthcare_VFL.__init__(self, args, dtype, 'MIMIC_VFL')


class PTBXL_VFL(Healthcare_VFL):
    def __init__(self, args, dtype):
        Healthcare_VFL.__init__(self, args, dtype, 'PTBXL_VFL')


class MUJOCO_VFL(Robotic_VFL):
    def __init__(self, args, dtype):
        Robotic_VFL.__init__(self, args, dtype)


class VISIONTOUCH_VFL(Dataset):
    def __init__(
            self,
            args,
            dtype,
            n_time_steps=1,
            pairing_tolerance=0.06,
    ):
        self.args = args
        with open('dataset/training_default.yaml') as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        filename_list = []
        for file in os.listdir(configs['dataset']):
            if file.endswith(".h5"):
                filename_list.append(configs['dataset'] + file)

        print(
            "Number of files in multifile dataset = {}".format(len(filename_list))
        )

        val_filename_list = []

        random.seed(args.seed)

        val_index = np.random.randint(
            0, len(filename_list), int(len(filename_list) * configs['val_ratio'])
        )

        for index in val_index:
            val_filename_list.append(filename_list[index])

        while val_index.size > 0:
            filename_list.pop(val_index[0])
            val_index = np.where(
                val_index > val_index[0], val_index - 1, val_index)
            val_index = val_index[1:]

        print("Initial finished")

        val_filename_list1, filename_list1 = augment_val(
            val_filename_list, filename_list
        )

        print("Listing finished")

        episode_length = configs['ep_length']
        training_type = configs['training_type']
        action_dim = configs['action_dim']

        if dtype == 'train':
            self.dataset_path = filename_list1
            self.missing_rate = args.p_miss_train
        elif dtype == 'valid':
            self.dataset_path = val_filename_list1
            self.missing_rate = args.p_miss_test

        self.transform = torchvision.transforms.Compose(
            [
                ProcessForce(32, "force", tanh=True),
                ProcessForce(32, "unpaired_force", tanh=True),
                ToTensor(),
                combine_modalitiesbuilder(unimodal=None, output='contact_next'),
            ]
        ),
        self.episode_length = episode_length
        self.training_type = training_type
        self.n_time_steps = n_time_steps
        self.dataset = {}
        self.action_dim = action_dim
        self.pairing_tolerance = pairing_tolerance

        self._config_checks()
        self._init_paired_filenames()

        self.data = []

        for idx in range(len(self.dataset_path) * (self.episode_length - self.n_time_steps)):
            """Get item in dataset at index idx."""
            list_index = idx // (self.episode_length - self.n_time_steps)
            dataset_index = idx % (self.episode_length - self.n_time_steps)
            filename = self.dataset_path[list_index][:-8]

            file_number, filename = self._parse_filename(filename)

            unpaired_filename, unpaired_idx = self.paired_filenames[(
                list_index, dataset_index)]

            if dataset_index >= self.episode_length - self.n_time_steps - 1:
                dataset_index = np.random.randint(
                    self.episode_length - self.n_time_steps - 1
                )

            sample = self._get_single(
                self.dataset_path[list_index],
                list_index,
                unpaired_filename,
                dataset_index,
                unpaired_idx,
            )

            self.data.append(sample)

        if self.args.eval_mode == 'robustness':
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
            distribution = get_vfl_data_distribution(self.args)
            if args.perturb_type == 'missing':
                pass
            elif args.perturb_type == 'corrupted':
                idxes_list = []
                for i in range(len(distribution)):
                    idxes_t = np.random.permutation(len(self.data))
                    idxes_t = idxes_t[0:perturb_data_len]
                    method_name = random.sample(d.keys(), 1)[0]
                    severity = np.random.randint(1, 5)
                    corruption = lambda clean_data: d[method_name](clean_data, severity)
                    for index in idxes_t:
                        self.data[index][distribution[i]] = corruption(self.data[index][distribution[i]])
            elif args.perturb_type == 'misaligned':
                list_misaligned_clients = list(range(0, self.args.client_num))
                list_misaligned_clients.remove(args.anchor_client)
                list_misaligned_data = []
                for client in iter(list_misaligned_clients):
                    misaligned_data = misaligned(np.array([self.data[i][distribution[client]] for i in idxes]))
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


        num_samples = len(self.data)
        num_batches = (num_samples + self.args.batch_size - 1) // self.args.batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(num_batches, self.args.client_num,
                                                                         self.missing_rate)
        self.batch_size = self.args.batch_size

    def __len__(self):
        """Get number of items in dataset."""
        return len(self.dataset_path) * (self.episode_length - self.n_time_steps)

    def __getitem__(self, idx):
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)

        if dataset_index >= self.episode_length - self.n_time_steps - 1:
            dataset_index = np.random.randint(
                self.episode_length - self.n_time_steps - 1
            )

        batch_idx = dataset_index // self.batch_size
        item_data = self.data[dataset_index]

        feature_img = item_data[0]
        feature_fce = item_data[1]
        feature_prio = item_data[2]
        feature_dep = item_data[3]
        feature_act = item_data[4]
        length = item_data[5]
        index = dataset_index
        label = item_data[7]

        mask = self.batch_patterns[batch_idx]

        if self.args.eval_mode == 'robustness':
            item_data_aug_1 = self.data_aug_1[dataset_index]
            item_data_aug_2 = self.data_aug_2[dataset_index]

            feature_img_aug_1 = item_data_aug_1[0]
            feature_fce_aug_1 = item_data_aug_1[1]
            feature_prio_aug_1 = item_data_aug_1[2]
            feature_dep_aug_1 = item_data_aug_1[3]
            feature_act_aug_1 = item_data_aug_1[4]

            feature_img_aug_2 = item_data_aug_2[0]
            feature_fce_aug_2 = item_data_aug_2[1]
            feature_prio_aug_2 = item_data_aug_2[2]
            feature_dep_aug_2 = item_data_aug_2[3]
            feature_act_aug_2 = item_data_aug_2[4]
            return [feature_img, feature_fce, feature_prio, feature_dep, feature_act, feature_img_aug_1, feature_fce_aug_1, feature_prio_aug_1, feature_dep_aug_1, feature_act_aug_1, feature_img_aug_2, feature_fce_aug_2, feature_prio_aug_2, feature_dep_aug_2, feature_act_aug_2, length, index, label, torch.tensor(mask, dtype=torch.bool)]
        return [feature_img, feature_fce, feature_prio, feature_dep, feature_act, length, index, label, torch.tensor(mask, dtype=torch.bool)]

    def _get_single(
            self, dataset_name, list_index, unpaired_filename, dataset_index, unpaired_idx
    ):

        dataset = h5py.File(dataset_name, "r", swmr=True, libver="latest")
        unpaired_dataset = h5py.File(
            unpaired_filename, "r", swmr=True, libver="latest")

        if self.training_type == "selfsupervised":

            image = dataset["image"][dataset_index]
            depth = dataset["depth_data"][dataset_index]
            proprio = dataset["proprio"][dataset_index][:8]
            force = dataset["ee_forces_continuous"][dataset_index]

            if image.shape[0] == 3:
                image = np.transpose(image, (2, 1, 0))

            if depth.ndim == 2:
                depth = depth.reshape((128, 128, 1))

            flow = np.array(dataset["optical_flow"][dataset_index])
            flow_mask = np.expand_dims(
                np.where(
                    flow.sum(axis=2) == 0,
                    np.zeros_like(flow.sum(axis=2)),
                    np.ones_like(flow.sum(axis=2)),
                ),
                2,
            )

            unpaired_image = image
            unpaired_depth = depth
            unpaired_proprio = unpaired_dataset["proprio"][unpaired_idx][:8]
            unpaired_force = unpaired_dataset["ee_forces_continuous"][unpaired_idx]

            sample = {
                "image": image,
                "depth": depth,
                "flow": flow,
                "flow_mask": flow_mask,
                "action": dataset["action"][dataset_index + 1],
                "force": force,
                "proprio": proprio,
                "ee_yaw_next": dataset["proprio"][dataset_index + 1][:self.action_dim],
                "contact_next": np.array(
                    [dataset["contact"][dataset_index + 1].sum() > 0]
                ).astype(float),
                "unpaired_image": unpaired_image,
                "unpaired_force": unpaired_force,
                "unpaired_proprio": unpaired_proprio,
                "unpaired_depth": unpaired_depth,
            }

        dataset.close()
        unpaired_dataset.close()

        if self.transform:
            sample = self.transform[0](sample)
        sample.insert(5, dataset_index)
        sample.insert(5, [])
        return sample

    def _init_paired_filenames(self):
        """
        Precalculates the paired filenames.
        Imposes a distance tolerance between paired images
        """
        tolerance = self.pairing_tolerance

        all_combos = set()

        self.paired_filenames = {}
        for list_index in tqdm(range(len(self.dataset_path)), desc="pairing_files"):
            filename = self.dataset_path[list_index]
            file_number, _ = self._parse_filename(filename[:-8])

            dataset = h5py.File(filename, "r", swmr=True, libver="latest")

            for idx in range(self.episode_length - self.n_time_steps):

                proprio_dist = None
                while proprio_dist is None or proprio_dist < tolerance:
                    # Get a random idx, file that is not the same as current
                    unpaired_dataset_idx = np.random.randint(self.__len__())
                    unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(
                        unpaired_dataset_idx)

                    while unpaired_filename == filename:
                        unpaired_dataset_idx = np.random.randint(
                            self.__len__())
                        unpaired_filename, unpaired_idx, _ = self._idx_to_filename_idx(
                            unpaired_dataset_idx)

                    with h5py.File(unpaired_filename, "r", swmr=True, libver="latest") as unpaired_dataset:
                        proprio_dist = np.linalg.norm(
                            dataset['proprio'][idx][:3] - unpaired_dataset['proprio'][unpaired_idx][:3])

                self.paired_filenames[(list_index, idx)] = (
                    unpaired_filename, unpaired_idx)
                all_combos.add((unpaired_filename, unpaired_idx))

            dataset.close()

    def _idx_to_filename_idx(self, idx):
        """
        Utility function for finding info about a dataset index

        Args:
            idx (int): Dataset index

        Returns:
            filename (string): Filename associated with dataset index
            dataset_index (int): Index of data within that file
            list_index (int): Index of data in filename list
        """
        list_index = idx // (self.episode_length - self.n_time_steps)
        dataset_index = idx % (self.episode_length - self.n_time_steps)
        filename = self.dataset_path[list_index]
        return filename, dataset_index, list_index

    def _parse_filename(self, filename):
        """ Parses the filename to get the file number and filename"""
        if filename[-2] == "_":
            file_number = int(filename[-1])
            filename = filename[:-1]
        else:
            file_number = int(filename[-2:])
            filename = filename[:-2]

        return file_number, filename

    def _config_checks(self):
        if self.training_type != "selfsupervised":
            raise ValueError(
                "Training type not supported: {}".format(self.training_type)
            )


class MMIMDB_VFL(Dataset):
    """Implements a torch Dataset class for the imdb dataset."""

    def __init__(self, args, dtype) -> None:
        self.args = args
        if dtype == 'train':
            start_ind = 0
            end_ind = 15552
        elif dtype == 'valid':
            start_ind = 15552
            end_ind = 18160
        elif dtype == 'test':
            start_ind = 18160
            end_ind = 25959
        self.file = args.data_dir + 'MMIMDB/multimodal_imdb.hdf5'
        self.start_ind = start_ind
        self.size = end_ind - start_ind
        self.vggfeature = True

        if not hasattr(self, 'dataset'):
            self.dataset = h5py.File(self.file, 'r')

        self.data = []

        for ind in range(self.size):
            text = self.dataset["features"][ind + self.start_ind]
            image = self.dataset["images"][ind + self.start_ind] if not self.vggfeature else \
                self.dataset["vgg_features"][ind + self.start_ind]
            label = self.dataset["genres"][ind + self.start_ind]
            data_item = [text, image, label]
            self.data.append(data_item)

        if args.eval_mode == 'robustness':
            import collections
            d = collections.OrderedDict()
            d['Gaussian Noise'] = gaussian_noise
            # d['Random Perturbation'] = random_perturb
            idxes = np.random.permutation(len(self.data))
            if dtype == 'train':
                perturb_data_len = int(args.perturb_rate_train * len(self.data))
                self.missing_rate = args.p_miss_train
            elif dtype == 'valid':
                perturb_data_len = int(args.perturb_rate_train * len(self.data))
                self.missing_rate = args.p_miss_train
            elif dtype == 'test':
                perturb_data_len = int(args.perturb_rate_test * len(self.data))
                self.missing_rate = args.p_miss_test
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
                    misaligned_data = misaligned(np.array([self.data[i][distribution[client]] for i in idxes]))
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

        num_samples = self.size
        num_batches = (num_samples + args.batch_size - 1) // args.batch_size
        self.batch_patterns, self.p_observed_array = _get_mask_per_batch(num_batches, args.client_num, self.missing_rate)
        self.batch_size = args.batch_size

    def __getitem__(self, ind):
        """Get item from dataset.

        Args:
            ind (int): Index of data to get

        Returns:
            tuple: Tuple of text input, image input, and label
        """
        batch_idx = ind // self.batch_size
        text = self.data[ind][0]
        image = self.data[ind][1]
        label = self.data[ind][2]

        length = []

        mask = self.batch_patterns[batch_idx]

        if self.args.eval_mode == 'robustness':
            text_aug_1 = self.data_aug_1[ind][0]
            image_aug_1 = self.data_aug_1[ind][1]

            text_aug_2 = self.data_aug_2[ind][0]
            image_aug_2 = self.data_aug_2[ind][1]
            return [text, image, text_aug_1, image_aug_1, text_aug_2, image_aug_2, length, ind, label, torch.tensor(mask, dtype=torch.bool)]
        return [text, image, length, ind, label, torch.tensor(mask, dtype=torch.bool)]

    def __len__(self):
        """Get length of dataset."""
        return self.size
