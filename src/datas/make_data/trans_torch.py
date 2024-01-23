from random import random

import torch
import numpy as np

from src.datas import FEATURE_TYPE, TARGET_TYPE
from src.library.cache_io import isfile_in_cache, load_file_in_cache, dump_file_in_cache
from src.library.logger import create_logger

logger = create_logger()


class StackedDataSet:
    """
    All data samples are stacked into a tensor `self.features`
    and `self.targets`
    """

    def __init__(self, features, targets):
        assert len(features) == len(targets)
        set_size = len(features)
        self.features = features
        self.targets = targets
        self.test_features = None
        self.test_targets = None

        # random shuffling
        self.__RR = False
        self.__order = list(range(set_size))

        self.set_size = set_size
        self.__COUNT = None

    def get_count(self):
        if self.__COUNT is None:
            count = {}
            for target_tensor in self.targets:
                target = target_tensor.item()
                if target in count.keys():
                    count[target] += 1
                else:
                    count[target] = 1
            self.__COUNT = count
        return self.__COUNT

    def random_reshuffle(self, on):
        self.__RR = on
        if on:
            random.shuffle(self.__order)

    def __getitem__(self, index):
        if self.__RR:
            i = self.__order[index]
            return self.features[i], self.targets[i]
        else:
            return self.features[index], self.targets[index]

    def __len__(self):
        return self.set_size

    def subset(self, indexes, name=''):
        if name == '':
            name = self.name + '_subset'
        return StackedDataSet(self.features[indexes], self.targets[indexes])

    def get_test_set(self):
        if self.test_features is None or self.test_targets is None:
            self.test_features = self.features
            self.test_targets = self.targets
        return StackedDataSet(self.test_features, self.test_targets)


class DataPackage:
    """
    A `DataPackage` contains the information, like the name and the data size,
    of a specific dataset (e.g. MNIST, CIFAR10, CIFAR100), and the references of
    train and test dataset.
    Therefore, the relation of DataPackage and Dataset is:
    DataPackage includes Dataset
    TODO: the arguments `train_set` and `test_set` can be other implementation
          of dataset
    """

    def __init__(self, name,
                 train_set: StackedDataSet,
                 test_set: StackedDataSet):
        self.name = name
        self.train_set = train_set
        self.test_set = test_set

        assert len(train_set) != 0, 'No data in train set'
        assert len(test_set) != 0, 'No data in test set'

        IDX_ARBITRARY_DATA_SAMPLE = 0
        IDX_FEATURE = 0
        train_sample_feature = train_set[IDX_ARBITRARY_DATA_SAMPLE][IDX_FEATURE]
        test_sample_feature = test_set[IDX_ARBITRARY_DATA_SAMPLE][IDX_FEATURE]

        assert train_sample_feature.nelement() == test_sample_feature.nelement()
        assert train_sample_feature.size() == test_sample_feature.size()

        self.feature_dimension = train_sample_feature.nelement()
        self.feature_size = train_sample_feature.size()

        logger.info("Data {} is loaded, the feature size is {}, the classes num is {}.".format(name,
                                                                                               self.feature_size,
                                                                                               self.num_classes))


class StackedTorchDataPackage(DataPackage):
    def __init__(self, name, load_fn, target_type=TARGET_TYPE):
        feature_dtype_str = str(FEATURE_TYPE).replace('torch.', '')
        target_dtype_str = str(target_type).replace('torch.', '')
        cache_file_name = f'data_cache_{name}_{feature_dtype_str}_{target_dtype_str}'
        if isfile_in_cache(cache_file_name):
            cache = load_file_in_cache(cache_file_name)
            train_features = cache['train_features']
            train_targets = cache['train_targets']
            test_features = cache['test_features']
            test_targets = cache['test_targets']
            num_classes = cache['num_classes']
        else:
            torch_train_set, torch_test_set = load_fn()
            train_features = torch.stack(
                [feature for feature, _ in torch_train_set], axis=0
            ).type(FEATURE_TYPE)
            test_features = torch.stack(
                [feature for feature, _ in torch_test_set], axis=0
            ).type(FEATURE_TYPE)
            train_targets = torch.Tensor(np.array(torch_train_set.targets)).type(target_type)
            test_targets = torch.Tensor(np.array(torch_test_set.targets)).type(target_type)
            num_classes = len(torch_train_set.classes)
            cache = {
                'train_features': train_features,
                'train_targets': train_targets,
                'test_features': test_features,
                'test_targets': test_targets,
                'num_classes': num_classes,
            }
            dump_file_in_cache(cache_file_name, cache)

        self.num_classes = num_classes
        train_set = StackedDataSet(train_features, train_targets)
        test_set = StackedDataSet(test_features, test_targets)
        super().__init__(name, train_set, test_set)

