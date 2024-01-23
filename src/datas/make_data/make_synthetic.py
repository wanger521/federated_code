import logging
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity

from src.datas.make_data import FEATURE_TYPE,  SYN_NUM_CLASSES, TARGET_TYPE_SYN, \
                                SYN_DATA_DISTRIBUTION_NUM  # These are defined in make_data/__init__.py
from src.datas.make_data.trans_torch import StackedTorchDataPackage


logger = logging.getLogger(__name__)


class SYNTHETIC(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "Synthetic"

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            num_classes: int = 1,
            syn_data_distribution_num: int = 1
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.num_classes = num_classes
        self.syn_num = syn_data_distribution_num if self.num_classes == 1 else num_classes

        self.DIMENSION_SYN = tuple((2, 2, 3))
        self.TRAIN_NUM_SYN = 1000 * self.syn_num
        self.TEST_NUM_SYN = 200 * self.syn_num

        self.train_list = [
            ["train_data_targetType_" + str(TARGET_TYPE_SYN).replace('torch.', '') + "_" + str(
                self.num_classes) + "_"
             + str(self.syn_num), None],
        ]

        self.test_list = [
            ["test_data_targetType_" + str(TARGET_TYPE_SYN).replace('torch.', '') + "_" + str(
                self.num_classes) + "_"
             + str(self.syn_num), None],
        ]
        self.meta = {
            "filename": "data_meta_targetType_" + str(TARGET_TYPE_SYN).replace('torch.', '') + "_" +
                        str(self.num_classes) + "_" + str(self.syn_num),
            "key": "label_names",
            "md5": None,
        }

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 2, 2)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        for filename, md5 in self.train_list + self.test_list:
            fpath = os.path.join(self.root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        """
        Create synthetic data in here!
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        train_class_each = self.TRAIN_NUM_SYN // self.syn_num
        test_class_each = self.TEST_NUM_SYN // self.syn_num
        image_train, label_train = np.empty(shape=(0,) + self.DIMENSION_SYN), np.empty(shape=(0, 1))
        image_test, label_test = np.empty(shape=(0,) + self.DIMENSION_SYN), np.empty(shape=(0, 1))
        for class_syn in range(self.syn_num):
            best_solution = np.random.normal(loc=class_syn, scale=1, size=self.DIMENSION_SYN)
            image_train_temp = np.random.normal(loc=class_syn, scale=1, size=(train_class_each,) + self.DIMENSION_SYN)
            image_test_temp = np.random.normal(loc=class_syn, scale=1, size=(test_class_each,) + self.DIMENSION_SYN)
            image_train = np.append(image_train, image_train_temp, axis=0)
            image_test = np.append(image_test, image_test_temp, axis=0)
            if "float" in str(TARGET_TYPE_SYN):  # continuous label
                label_train = np.append(label_train, (np.dot(image_train_temp.T.reshape(-1, 12),
                                                             best_solution.reshape(12, -1)).T + np.random.normal(loc=0,
                                                                                                                 scale=0.2,
                                                                                                                 size=train_class_each)).reshape(
                    -1, 1), axis=0)
                label_test = np.append(label_test, (np.dot(image_test_temp.T.reshape(-1, 12),
                                                           best_solution.reshape(12, 1)).T + np.random.normal(loc=0,
                                                                                                              scale=0.2,
                                                                                                              size=test_class_each)).reshape(
                    -1, 1), axis=0)
            else:  # class label
                label_train = np.append(label_train, np.ones(shape=(train_class_each, 1)) * class_syn, axis=0)
                label_test = np.append(label_test, np.ones(shape=(test_class_each, 1)) * class_syn, axis=0)

        train = {"data": image_train, "labels": label_train}
        test = {"data": image_test, "labels": label_test}
        temp = [train, test]
        for count, filename in enumerate(self.train_list + self.test_list):
            fpath = os.path.join(self.root, self.base_folder, filename[0])
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            # np.save(fpath, temp[count])
            with open(fpath, "wb") as f:
                pickle.dump(temp[count], f)

        metas = {}
        class_mata = []
        for i in range(self.num_classes):
            class_mata.append(str(i))
        metas["label_names"] = class_mata
        metas["feature_type"] = FEATURE_TYPE
        metas["target_type"] = TARGET_TYPE_SYN
        metas["data_type"] = "continuous" if self.num_classes == 1 else "classified"
        metas["num_classes"] = self.num_classes
        metas["syn_data_distribution_num"] = self.syn_num
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        with open(path, "wb") as f:
            pickle.dump(metas, f)

        print("Data created!")
        logger.info("Data created, the meta data of synthetic is {}.".format(metas))

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


def get_synthetic(num_classes=SYN_NUM_CLASSES, syn_data_distribution_num=SYN_DATA_DISTRIBUTION_NUM):
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        # transforms.RandomCrop(32, padding=4),  # Fill all sides with 0, then crop the image randomly to 32*32
        # # 先四周填充0，再把图像随机裁剪成32*32
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean, std),
        # transforms.Lambda(lambda x: x / 255)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        # transforms.Normalize(mean, std),
    ])

    root = 'data/raw'
    torch_train_dataset = SYNTHETIC(root=root, train=True,
                                    transform=transform_train, download=True, num_classes=num_classes,
                                    syn_data_distribution_num=syn_data_distribution_num)
    torch_test_dataset = SYNTHETIC(root=root, train=False,
                                   transform=transform_test, download=True, num_classes=num_classes,
                                   syn_data_distribution_num=syn_data_distribution_num)
    return torch_train_dataset, torch_test_dataset


class MakeSynthetic(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('synthetic{}_{}'.format(SYN_NUM_CLASSES, SYN_DATA_DISTRIBUTION_NUM), get_synthetic,
                         TARGET_TYPE_SYN)
        # Use the TARGET_TYPE_SYN parameter "target_type" change the target type
