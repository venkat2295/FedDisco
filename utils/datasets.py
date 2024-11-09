import os
import os.path
import logging
import numpy as np
import torch
import pandas as pd
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, DatasetFolder
# Change this import to be relative
from heart_dataset_loader import prepare_heart_dataset  # Note the dot before heart_dataset_loader

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

class HeartDataset_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Download dataset if needed
        if download and not os.path.exists(os.path.join(root, 'train.csv')):
            from heart_dataset_loader import prepare_heart_dataset
            prepare_heart_dataset(root)

        # Load either train or test set
        file_path = os.path.join(root, 'train.csv' if train else 'test.csv')
        self.data, self.target = self.__build_truncated_dataset__(file_path)

        # Get the number of features for model initialization
        self.n_features = self.data.shape[1]

    def __build_truncated_dataset__(self, file_path):
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Split features and target
        features = df.iloc[:, :-1].values.astype(np.float32)
        targets = df.iloc[:, -1].values.astype(np.int64)

        if self.dataidxs is not None:
            features = features[self.dataidxs]
            targets = targets[self.dataidxs]

        return features, targets

    def __getitem__(self, index):
        features, target = self.data[index], self.target[index]

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return torch.FloatTensor(features), target

    def __len__(self):
        return len(self.data)
def partition_heart_data(dataset, num_users):
    """
    Partition the dataset for federated learning.
    Simulates realistic scenario where different hospitals have different patient distributions.
    """
    # Get the data and labels
    data = dataset.data
    labels = dataset.target

    # Sort data by label
    label_indices = {i: np.where(labels == i)[0] for i in np.unique(labels)}

    # Determine number of samples per user
    num_samples = int(len(dataset) / num_users)

    # Create imbalanced distribution
    partition_indices = []
    for i in range(num_users):
        user_indices = []
        # Biased sampling to create natural imbalance
        for label in np.unique(labels):
            # Create different proportions for different users
            prop = np.random.beta(2, 2)  # Beta distribution for natural variation
            label_count = int(prop * num_samples / len(np.unique(labels)))
            if len(label_indices[label]) >= label_count:
                selected_indices = np.random.choice(label_indices[label], label_count, replace=False)
                user_indices.extend(selected_indices)
                label_indices[label] = np.setdiff1d(label_indices[label], selected_indices)

        partition_indices.append(user_indices)

    return partition_indices


class CIFAR10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)
        data = cifar_dataobj.data
        target = np.array(cifar_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        fmnist_dataobj = FashionMNIST(self.root, self.train, None, None, self.download)
        data = np.array(fmnist_dataobj.data)
        target = np.array(fmnist_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class ImageFolder_CINIC10(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)
            
    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)


class ImageFolder_HAM10000(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)