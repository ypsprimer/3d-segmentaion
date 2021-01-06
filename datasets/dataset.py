import os
import time
import warnings
import typing
import torch
import copy

import numpy as np

from scipy.ndimage import affine_transform
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import List


def get_dataset(config: dict, phase: str) -> Dataset:
    """
    This function return a dataset according to the number of splits
    :param config:
    :param phase:
    :return:
    """
    train_split = config["prepare"]["train_split"]
    val_split = config["prepare"]["val_split"]
    all_split = config["prepare"]["all_split"]

    if isinstance(train_split, list):
        assert (
            len(train_split) == len(val_split) == len(all_split)
        ), "Lengths of train split, validation split, and all split should be the same."
        return MixTrainingDataset(config=config, phase=phase)
    elif isinstance(train_split, str):
        return dataset(config=config, phase=phase)


def get_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = False,
    drop_last: bool = True,
    collate_fn=None,
) -> DataLoader:

    if isinstance(dataset, MixTrainingDataset):
        return DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=MixTrainingSampler(data_source=dataset, batch_size=batch_size),
            collate_fn=mix_training_collect_fn
        )
    else:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )


class dataset(Dataset):
    def __init__(self, config, phase):

        assert phase in ["train", "val", "test"]
        self.phase = phase
        self.config = config

        if phase == "train":
            split = config["prepare"]["train_split"]
        elif phase == "val":
            split = config["prepare"]["val_split"]
        elif phase == "test":
            split = config["prepare"]["all_split"]
        else:
            raise NotImplementedError("Invalid phase")

        with open(split, "r") as f:
            self.cases = f.readlines()

        self.cases = [f.split("\n")[0] for f in self.cases]  #### 根据对应的类型，取出所有的数据编号
        datadir = config["prepare"]["data_dir"]

        self.img_files = [
            os.path.join(datadir, f + config["prepare"]["img_subfix"])
            for f in self.cases
        ]

        # not necessary to load label when test
        if self.phase != "test":
            self.lab_files = [
                os.path.join(datadir, f + config["prepare"]["lab_subfix"])
                for f in self.cases
            ]

        self.augmentation = augmentation(config)

    def __getitem__(self, idx):
        idx = idx % len(self.cases)

        if self.phase == "train":
            img, lab = self.getraw(idx)
            img, lab = self.augmentation.process(img, lab)
            img = self.__lum_trans__(img)
            return img[np.newaxis, :], lab[np.newaxis, :], self.cases[idx]

        if self.phase == "val":
            img, lab = self.getraw(idx)
            img = self.__lum_trans__(img)
            return img[np.newaxis, :], lab[np.newaxis, :], self.cases[idx]

        if self.phase == "test":
            img, fake_lab = self.getraw(idx)
            img = self.__lum_trans__(img)
            return img[np.newaxis, :], fake_lab, self.cases[idx]

    def getraw(self, idx: int) -> typing.Union[List[np.ndarray], np.ndarray]:
        img = np.load(self.img_files[idx]).astype(np.float)
        if self.phase != "test":
            lab = np.load(self.lab_files[idx])
            return [img, lab]
        else:
            return [img, np.zeros_like(img)]

    def __lum_trans__(self, x: np.ndarray):
        x = x.astype(np.float)
        if self.config["prepare"]["clip"]:
            x = np.clip(
                x, self.config["prepare"]["lower"], self.config["prepare"]["higher"]
            )
        # if self.config["prepare"]["normalize"]:
        #     mean = self.config["prepare"]["sub_value"]
        #     std = self.config["prepare"]["div_value"]
        #     x = (x.astype("float32") - mean) / std
        return x

    def __len__(self):
        if self.phase == "train":
            return len(self.cases) * self.config["train"]["train_repeat"]
        else:
            return len(self.cases)


class augmentation:
    def __init__(self, config):
        augtype = config["augtype"]

        self.swap = augtype["swap_axis"] if augtype["swap"] is True else None
        self.flip = augtype["flip_axis"] if augtype["flip"] is True else None
        self.scale = augtype["scale_lim"] if augtype["scale"] is True else None
        self.rotate = augtype["rotate_deg_lim"] if augtype["rotate"] is True else None

        self.crop_size = config["prepare"]["crop_size"]
        self.rng = np.random.default_rng()

    def process(self, img: np.ndarray, lab: np.ndarray = None):

        if self.swap is not None:
            swap_axis = np.nonzero(self.swap)[0]
            np.random.shuffle(swap_axis)
            img = img.swapaxes(swap_axis[0], swap_axis[1])
            lab = lab.swapaxes(swap_axis[0], swap_axis[1])

        if self.flip is not None:
            random = np.random.binomial(n=1, p=0.5, size=[2])
            img = np.flip(img, np.nonzero(random == 1)[0])
            lab = np.flip(lab, np.nonzero(random == 1)[0])

        transformation = np.zeros(shape=[3, 4])
        transformation[0, 0] = 1
        transformation[1, 1] = 1
        transformation[2, 2] = 1

        if self.scale is not None:
            scale_yx = np.random.uniform(self.scale[0], self.scale[1])
            shift_yx = (img.shape[2] - scale_yx * img.shape[2]) / 2
        else:
            scale_yx = 1
            shift_yx = 0

        if self.rotate is not None:

            rotate_z = np.random.uniform(low=0, high=self.rotate - 1) / 180 * np.pi
            shift_centroid = (
                img.shape[2]
                / 2
                * (
                    np.array([1, 1])
                    - np.array(
                        [
                            np.cos(rotate_z) + np.sin(rotate_z),
                            np.cos(rotate_z) - np.sin(rotate_z),
                        ]
                    )
                )
            )
        else:
            rotate_z = 0
            shift_centroid = np.array([0, 0])

        transformation[1, 1] = +scale_yx * np.cos(rotate_z)
        transformation[1, 2] = -scale_yx * np.sin(rotate_z)
        transformation[2, 1] = +scale_yx * np.sin(rotate_z)
        transformation[2, 2] = +scale_yx * np.cos(rotate_z)
        transformation[1, 3] = shift_yx + shift_centroid[1]
        transformation[2, 3] = shift_yx + shift_centroid[0]

        img = affine_transform(img, transformation, output_shape=img.shape)
        lab = affine_transform(lab, transformation, output_shape=lab.shape, order=0)

        pad_or_crop = (
            np.array(img.shape) - self.crop_size
        )  # Positive is cropping, Negative is padding
        random_crop = self.rng.integers(
            low=0,
            high=np.amax([np.array([0, 0, 0]), pad_or_crop], axis=0) + 1,
            size=(3),
        )

        pad_lim = np.amax([np.array([0, 0, 0]), -pad_or_crop], axis=0)
        random_pad = self.rng.integers(low=0, high=pad_lim + 1, size=(3),)

        if np.any(pad_or_crop < 0):
            img = np.pad(
                img,
                pad_width=(
                    (random_pad[0], pad_lim[0] - random_pad[0]),
                    (random_pad[1], pad_lim[1] - random_pad[1]),
                    (random_pad[2], pad_lim[2] - random_pad[2]),
                ),
                mode="constant",
            )
            lab = np.pad(
                lab,
                pad_width=(
                    (random_pad[0], pad_lim[0] - random_pad[0]),
                    (random_pad[1], pad_lim[1] - random_pad[1]),
                    (random_pad[2], pad_lim[2] - random_pad[2]),
                ),
                mode="constant",
            )

        if np.any(pad_or_crop > 0):
            img = img[
                random_crop[0] : random_crop[0] + self.crop_size[0],
                random_crop[1] : random_crop[1] + self.crop_size[1],
                random_crop[2] : random_crop[2] + self.crop_size[2],
            ]
            lab = lab[
                random_crop[0] : random_crop[0] + self.crop_size[0],
                random_crop[1] : random_crop[1] + self.crop_size[1],
                random_crop[2] : random_crop[2] + self.crop_size[2],
            ]

        return img, lab


class MixTrainingDataset(Dataset):
    def __init__(self, config: dict, phase: str):
        train_splits = config["prepare"]["train_split"]
        val_splits = config["prepare"]["val_split"]
        all_splits = config["prepare"]["all_split"]
        data_dirs = config["prepare"]["data_dir"]

        self.phase = phase
        self.datasets = []
        self.idx_sequence_dict = dict()
        self.idx_sequence_mapping = list()
        sample_count = 0
        for idx, (train_split, val_split, all_split, data_dir) in enumerate(
            zip(train_splits, val_splits, all_splits, data_dirs)
        ):
            sub_config = copy.deepcopy(config)
            sub_config["prepare"]["data_dir"] = data_dir
            sub_config["prepare"]["train_split"] = train_split
            sub_config["prepare"]["val_split"] = val_split
            sub_config["prepare"]["all_split"] = all_split
            self.datasets.append(dataset(config=sub_config, phase=phase),)
            self.idx_sequence_mapping.append([])
            for i in range(len(self.datasets[-1])):
                self.idx_sequence_mapping[-1].append(sample_count)
                self.idx_sequence_dict[sample_count] = idx
                sample_count += 1
            self.idx_sequence_mapping[-1] = np.array(self.idx_sequence_mapping[-1])

    def __len__(self):

        return len(self.idx_sequence_dict)

    def __getitem__(self, idx: int):

        dataset_belongs = self.idx_sequence_dict[idx]
        return (dataset_belongs, self.datasets[dataset_belongs][idx])


class MixTrainingSampler(Sampler):
    def __init__(self, data_source: MixTrainingDataset, batch_size: int):
        super(MixTrainingSampler).__init__()

        self.batch_size = batch_size
        self.idx_dataset_mapping = data_source.idx_sequence_mapping
        self.phase = data_source.phase
        self.length = 0
        for sequence in self.idx_dataset_mapping:
            self.length += len(sequence) // batch_size

    def __iter__(self):

        current_batch = []
        for i in range(len(self.idx_dataset_mapping)):
            np.random.shuffle(self.idx_dataset_mapping[i])
            drop_last_seq = self.idx_dataset_mapping[i][:len(self.idx_dataset_mapping[i]) // self.batch_size * self.batch_size]

            current_batch.append(np.reshape(drop_last_seq, newshape=(-1, self.batch_size)))

        current_batch = np.concatenate(current_batch, axis=0)

        return iter(current_batch.tolist())

    def __len__(self):
        return self.length


def mix_training_collect_fn(batch_data):
    sequence_idx = []
    imgs = []
    labs = []
    case_id = []

    for single_data in batch_data:
        sequence_idx.append(single_data[0])
        imgs.append(single_data[1][0])
        labs.append(single_data[1][1])
        case_id.append(single_data[1][2])

    sequence_idx = np.array(sequence_idx)
    imgs = torch.tensor(imgs)
    labs = torch.tensor(labs)
    case_id = tuple(case_id)
    assert np.all(sequence_idx == sequence_idx[0])

    return imgs, labs, case_id, sequence_idx[0]