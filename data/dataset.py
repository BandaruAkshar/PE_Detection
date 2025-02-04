import os
import cv2
import torch
import numpy as np

from tqdm.notebook import tqdm
from torch.utils.data import Dataset

from params import IMG_PATH, FEATURES_PATH, IMG_TARGET, TARGETS, MAX_LEN


def sop_to_image(sop, folder):
    files = os.listdir(folder)
    for image in files:
        if image.endswith(sop + ".jpg"):
            return image
    print("Not Found")
    return np.random.choice(files)


def get_image_name(df):
    image_names = []
    studies = df["StudyInstanceUID"].values
    series = df["SeriesInstanceUID"].values
    sops = df["SOPInstanceUID"].values
    for idx in tqdm(range(len(df))):
        folder = IMG_PATH + "/" + studies[idx] + "/" + series[idx] + "/"
        image_name = sop_to_image(sops[idx], folder)
        image_names.append(image_name)

    return image_names


class PEDatasetImg(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms

        self.patients = [
            p + "_" + z for p, z in df[["StudyInstanceUID", "SeriesInstanceUID"]].values
        ]
        self.image_paths = df["img_path"].values
        self.targets = self.df[TARGETS].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])

        if self.transforms:
            image = self.transforms(image=image)["image"]

        y = torch.tensor(self.targets[idx], dtype=torch.float)
        return image, y


class PatientDataset(Dataset):

    def __int__(self, path, transforms=None):
        self.path = path
        self.img_paths = sorted(os.listdir(path))
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.path + self.img_paths[idx])

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, idx


class PEDatasetFeatures(Dataset):
    def __init__(self, df, paths=[FEATURES_PATH], max_len=MAX_LEN):
        self.df = df
        self.paths = [[path + p for path in paths] for p in self.df["path"].values]
        self.max_len = max_len

        self.image_targets = df[IMG_TARGET].values
        self.targets = df[TARGETS].values

        self.image_targets = []
        for t in df[IMG_TARGET].values:
            self.image_targets.append(self.pad(np.array(t)))
        self.image_targets = np.array(self.image_targets)

    def __len__(self):
        return len(self.df)

    def pad(self, x):
        length = x.shape[0]
        if length > self.max_len:
            return x[:self.max_len]
        else:
            padded = np.zeros([self.max_len] + list(x.shape[1:]))
            padded[:length] = x
            return padded

    def __getitem__(self, idx):
        features = np.concatenate([np.load(p) for p in self.paths[idx]], -1)
        size = min(features.shape[0], self.max_len)
        features = self.pad(features)

        return (
            torch.tensor(features, dtype=torch.float),
            torch.tensor(self.targets[idx], dtype=torch.float),
            torch.tensor(self.image_targets[idx], dtype=torch.float),
            torch.tensor(size, dtype=torch.int64),
        )
