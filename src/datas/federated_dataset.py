import importlib
import json
import logging
import os
from functools import partial
from importlib import import_module
import random
import torch

from src.datas.partition import partition_continuous, partition_classification
from src.library.logger import create_logger

logger = create_logger()
SYNTHETIC = "Synthetic"


def random_generator(dataset, batch_size=1):
    lens = len(dataset)
    while True:
        beg = random.randint(0, lens - 1)
        if beg + batch_size <= lens:
            yield dataset[beg:beg + batch_size]
        else:
            features, targets = zip(dataset[beg:beg + batch_size],
                                    dataset[0:(beg + batch_size) % lens])
            yield torch.cat(features), torch.cat(targets)


def order_circle_generator(dataset, batch_size=1, *args, **kwargs):
    """In order to generate the data, ends when data traversal is complete. """
    lens = len(dataset)
    beg = 0
    while beg < lens-1:
        end = min(beg + batch_size, lens)
        yield dataset[beg:end]
        if end == lens:
            beg = 0
        else:
            beg += batch_size


def order_generator(dataset, batch_size=1):
    lens = len(dataset)
    beg = 0
    while beg < lens-1:
        end = min(beg + batch_size, lens)
        yield dataset[beg:end]


def full_generator(dataset, batch_size=1):
    while True:
        yield dataset[:]


def byrd_saga_generator(dataset, batch_size=1, *args, **kwargs):
    """The generator for byrd_saga applications. """
    lens = len(dataset)
    beg = 0
    while beg < lens-1:
        end = min(beg + batch_size, lens)
        yield dataset[beg:end], beg
        if end + batch_size >= lens:
            beg = 0
        else:
            beg += batch_size


def customized_generator(dataset, batch_size=1, *args, **kwargs):
    """
    Define your own data generator.
    """
    pass


def load_data(dataset_name):
    module_data = import_module("src.datas.make_data.make_{}".format(dataset_name.lower()))
    module_class = getattr(module_data, "Make{}".format(dataset_name))
    data = module_class()
    return data


def construct_partition_datasets(
        dataset_name,
        num_of_par_nodes,
        partition_type,
        test_partition_type,
        class_per_node,
        min_size,
        alpha):
    data = load_data(dataset_name)
    extra_info = {"num_classes": data.num_classes, "feature_dimension": data.feature_dimension,
                  "feature_size": data.feature_size}

    # load partition
    partition_type_dict_class = {"iid": "IIDPartition", "successive": "SuccessivePartition",
                                 "noniid_class": "NonIIDSeparation", "noniid_class_unbalance": "NonIIDSeparation",
                                 "noniid_dir": "DirichletNonIID",
                                 "share": "SharedData", "empty": "EmptyPartition"
                                 }
    partition_type_dict_continuous = {"iid": "IIDPartition", "successive": "SuccessivePartition",
                                      "noniid_class": "SuccessivePartition",
                                      "noniid_class_unbalance": "SuccessivePartition",
                                      "noniid_dir": "SuccessivePartition",
                                      "share": "SharedData", "empty": "EmptyPartition"
                                      }

    if dataset_name == SYNTHETIC and data.num_classes == 1:
        partition_train_load = getattr(partition_continuous, partition_type_dict_continuous[partition_type])
        partition_test_load = getattr(partition_continuous, partition_type_dict_continuous[test_partition_type])
    else:
        partition_train_load = getattr(partition_classification, partition_type_dict_class[partition_type])
        partition_test_load = getattr(partition_classification, partition_type_dict_class[test_partition_type])

    data_balance = False if partition_type == "noniid_class_unbalance" else True
    train_partition = partition_train_load(dataset=data.train_set, node_cnt=num_of_par_nodes,
                                           class_per_node=class_per_node, data_balance=data_balance,
                                           alpha=alpha, min_size=min_size)
    # train_partition.print_data_distribution()
    for x in train_partition:
        random.shuffle(x)
    data_balance = False if test_partition_type == "noniid_class_unbalance" else True
    test_partition = partition_test_load(dataset=data.test_set, node_cnt=num_of_par_nodes,
                                         class_per_node=class_per_node, data_balance=data_balance, alpha=alpha,
                                         min_size=min_size)
    # TODO: There will be a lot of duplicate data under full information, and each node has full datas.
    for x in test_partition:
        random.shuffle(x)
    train_par_set = train_partition.get_subsets(data.train_set)
    test_par_set = test_partition.get_subsets(data.test_set)
    return train_par_set, test_par_set, extra_info


def construct_federated_datasets(dataset_name,
                                 num_of_par_nodes,
                                 partition_type,
                                 test_partition_type,
                                 class_per_node,
                                 min_size,
                                 alpha,
                                 generator,
                                 train_batch_size,
                                 test_rate):
    """Simulate and load federated datasets.

    Args:
        dataset_name (str): The name of the dataset. It currently supports: cifar10, cifar100, mnis and synthetic.
        num_of_par_nodes (int): The targeted number of nodes to construct.
        partition_type (str): The type of statistical simulation, options: iid, successive, empty, share, noniid_class,
                                                                                    noniid_class_unbalance, noniid_dir.
            `iid` means independent and identically distributed data.
            `successive` means continuous equal scaling of the data.
            `empty` means empty dataset for each node.
            `share`means each node has full dataset.
            `noniid_class` means non-independent and identically distributed data, means partitioning the dataset
                            by label classes, for classified datasets.
            `noniid_class_unbalance` means non-independent and identically distributed data, means partitioning the dataset
                            by label classes but unbalanced, each node may have big different size of dataset, for classified datasets.
            `noniid_dir` means using Dirichlet process to simulate non-iid data, for classified datasets.
        test_partition_type (str): Similar to  train_partition_type, default `share`, other partition type can also be set
        class_per_node (int): The number of classes in each node. Only applicable when the partition_type is `noniid_class`
                                                                                        and `noniid_class_unbalance`.
        min_size (int): The minimal number of samples in each node.
                        It is applicable for `noniid_dir` partition of classified dataset.
        alpha (float): The unbalanced parameter of samples in each node. Default 0.1.
                        It is applicable for `noniid_dir` partition of classified dataset.

        generator (str): The name of iter generator, including random, order, order_circle, full.
        train_batch_size (int): Data batch needed for one computation during the training process.
        test_rate (float): (0, 1], test dataset of the local node of the scale used for the test.
    Returns:
        train_data_iter, test_data_iter
    """

    train_par_set, test_par_set, extra_info = construct_partition_datasets(dataset_name=dataset_name,
                                                                           num_of_par_nodes=num_of_par_nodes,
                                                                           partition_type=partition_type,
                                                                           test_partition_type=test_partition_type,
                                                                           class_per_node=class_per_node,
                                                                           min_size=min_size,
                                                                           alpha=alpha)

    extra_info["train_data_size_each"] = [len(x) for x in train_par_set]
    extra_info["test_data_size_each"] = [len(x) for x in test_par_set]
    # load partition dataset iter
    data_generator = eval("{}_generator".format(generator))
    get_train_iter = partial(data_generator,
                             batch_size=train_batch_size)
    train_data_iter = [get_train_iter(dataset=train_par_set[node]) for node in range(num_of_par_nodes)]

    test_generator = full_generator if test_rate == 1 else order_circle_generator
    # test_generator = full_generator if test_rate == 1 else random_generator
    get_test_iter = partial(test_generator)
    test_data_iter = [get_test_iter(dataset=test_par_set[node], batch_size=int(len(test_par_set[node]) * test_rate))
                      for node in range(num_of_par_nodes)]

    return train_data_iter, test_data_iter, extra_info
