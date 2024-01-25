import logging
import os
from copy import copy
from importlib import import_module

import torch

import src
from src.datas.federated_dataset import construct_federated_datasets
from src.datas.make_data import *
from src.datas.partition import partition_classification as partition, partition_continuous
from src.coordinate import init_conf, init_logger, _set_random_seed
from src.library import graph
from src.library.graph import CompleteGraph, ErdosRenyi, TwoCastle, RingCastle, OctopusGraph
from src.models import load_model
from src.library.logger import create_logger
from src.tracking import metric
import wandb
import random

REQUIRED_PYTHON = "python3"


def main():
    # test_partition_class()
    # test_partition_continuous()
    # test_models()
    # test_module()
    # test_coordinate()
    # test_graph()
    # test_online()
    test_wandb()


def test_conf():
    confs = init_conf()
    init_logger(logging.INFO)
    _set_random_seed(confs.seed)
    print(confs)


def test_partition_class():
    # x = make_mnist.MakeMnist().test_set
    # y = make_cifar100.MakeCifar100().train_set
    y = make_cifar10.MakeCifar10().train_set
    # par = partition.IIDPartition(y, 12)
    # par = partition.NonIIDSeparation(y, 5)
    # par = partition.NonIIDSeparation(dataset=y, node_cnt=5, class_per_node=2, data_balance=False)
    # par = partition.DirichletNonIID(dataset=y, node_cnt=5, alpha=0.1, min_size=10)
    # par = partition.VerticalPartition(dataset=y, node_cnt=5)

    # par = partition.SuccessivePartition(dataset=y, node_cnt=12)
    # par = partition.LabelSeparation(dataset=y, node_cnt=12)
    # par = partition.NonIIDSeparation(dataset=y, node_cnt=5, class_per_node=2, data_balance=True)
    # par = partition.NonIIDSeparation(dataset=y, node_cnt=20, class_per_node=6, data_balance=False)
    par = partition.DirichletNonIID(dataset=y, node_cnt=10, alpha=0.2, min_size=10)

    # distribution = par.data_distribution
    # print(distribution)
    par.draw_data_distribution()
    print(y.__len__())
    return par


def test_partition_continuous():
    y = make_synthetic.MakeSynthetic().train_set
    # par = partition_continuous.IIDPartition(y, 5)
    par = partition_continuous.SuccessivePartition(y, 10)
    # par = partition_continuous.SharedData(y, 5)
    par.draw_data_distribution()
    print(y.__len__())
    return par


def test_models():
    # model = load_model("resnet18")(num_classes=10)
    model = load_model("linear_regression")(num_classes=1)
    x = torch.Tensor([[1], [2], [3], [4]])
    out = model(x)
    print(out)


def test_module():
    # dataset_name = "Cifar10"  # "Synthetic"
    # train_data_iter, test_data_iter,_ = construct_federated_datasets(dataset_name="Synthetic",
    #                                                                num_of_par_nodes=5,
    #                                                                partition_type="iid",
    #                                                                test_partition_type="share",
    #                                                                class_per_node=4,
    #                                                                min_size=10,
    #                                                                alpha=0.1,
    #                                                                generator="random",
    #                                                                train_batch_size=2,
    #                                                                test_rate=1)
    # train_data_iter, test_data_iter,_ = construct_federated_datasets(dataset_name="Cifar10",
    #                                                                num_of_par_nodes=10,
    #                                                                partition_type="noniid_class",
    #                                                                test_partition_type="noniid_class",
    #                                                                class_per_node=4,
    #                                                                min_size=10,
    #                                                                alpha=0.1,
    #                                                                generator="random",
    #                                                                train_batch_size=2,
    #                                                                test_rate=0.001)
    train_data_iter, test_data_iter, extra_info = construct_federated_datasets(dataset_name="Mnist",
                                                                               num_of_par_nodes=10,
                                                                               partition_type="successive",
                                                                               test_partition_type="share",
                                                                               class_per_node=4,
                                                                               min_size=10,
                                                                               alpha=0.1,
                                                                               generator="random",
                                                                               train_batch_size=64,
                                                                               test_rate=0.01)
    x = next(train_data_iter[0])
    y = next(test_data_iter[0])
    print(extra_info)
    print(x[1])

    print(y[1])


def test_coordinate():
    # config = {
    #     "data": {"dataset": "Cifar10", "partition_type": "noniid_class", "num_of_par_nodes": 2},
    #     "model": "simple_cnn"
    # }
    # config = {
    #     "data": {"dataset": "Synthetic", "partition_type": "iid", "num_of_par_nodes": 5},
    #     "model": "linear_regression"
    # }
    config = {
        "data": {"dataset": "Mnist", "partition_type": "iid"},
        "model": "softmax_regression",
        "gpu": 0,
        "node": {"epoch_or_iteration": "iteration", "message_type_of_node_sent": "model", "local_epoch": 10,
                 "optimizer": {"type": "SGD", "use_another_momentum": True}, "local_iteration": 1,
                 "lr_controller": "ConstantLr",
                 "track": True},
        "graph": {"centralized": True, "graph_type": "CompleteGraph", "byzantine_cnt": 2, "nodes_cnt": 10},
        "controller": {"nodes_per_round": 10, "random_selection": False,
                       "aggregation_rule": "Mean", "attack_type": "NoAttack", "rounds_iterations": 1000},
        "aggregation_param": {"exact_byz_cnt": True, "byz_cnt": 0, "weight_mh": True,
                              "threshold_selection": "true", "threshold": 10},
        "attacks_param": {"use_honest_mean": True, "sign_scale": -4, "little_scale": None}
    }
    src.init(config)
    src.run()


def test_online():
    config = {
        "data": {"dataset": "Mnist", "partition_type": "iid"},
        "model": "softmax_regression",
        "gpu": 0,
        "node": {"epoch_or_iteration": "iteration", "message_type_of_node_sent": "model", "local_epoch": 2,
                 "optimizer": {"type": "SGD", "use_another_momentum": True}, "local_iteration": 1,
                 "lr_controller": "ConstantLr"},
        "graph": {"centralized": False, "graph_type": "CompleteGraph", "byzantine_cnt": 2, "nodes_cnt": 10},
        "controller": {"nodes_per_round": 10, "random_selection": False,
                       "aggregation_rule": "Mean", "attack_type": "NoAttack", "rounds_iterations": 1000},
        "lr_controller_param": {"init_lr": 0.1, "init_momentum": 0.1},
        "aggregation_param": {"exact_byz_cnt": True, "byz_cnt": 0, "weight_mh": True,
                              "threshold_selection": "true", "threshold": 10},
        "attacks_param": {"use_honest_mean": False, "sign_scale": -4, "little_scale": None}
    }

    if "node" in config:
        config["node"]["calculate_static_regret"] = True
    else:
        config["node"] = {"calculate_static_regret": True}

    config_online = copy(config)
    config_online["task_name"] = metric.ONLINE_BEST_MODEL
    config_online["data"] = {"partition_type": "iid"}
    config_online["controller"] = {"nodes_per_round": 1, "random_selection": False,
                                   "aggregation_rule": "Mean", "attack_type": "NoAttack"}
    if "rounds" in config["controller"]:
        config_online["controller"]["rounds"] = 5 * config["controller"]["rounds"]
    else:
        config_online["controller"]["rounds"] = 50
    if "rounds_iterations" in config["controller"]:
        config_online["controller"]["rounds_iterations"] = 5 * config["controller"]["rounds_iterations"]
    else:
        config_online["rounds_iterations"] = 25000
    config_online["controller"]["save_model_every_epoch"] = config_online["controller"]["rounds"]
    config_online["controller"]["save_model_every_iteration"] = config_online["controller"]["rounds_iterations"]
    config_online["graph"] = {"centralized": True, "nodes_cnt": 1, "byzantine_cnt": 0}

    src.init(config_online)
    src.run()

    src.init(config)
    src.run()


def test_graph():
    # graph = CompleteGraph(node_size=10, byzantine_size=2)
    # graph = ErdosRenyi(node_size=10, byzantine_size=2,connected_p=0.7)
    # graph = TwoCastle(node_size=10, byzantine_size=1)
    # graph = RingCastle(node_size=10, byzantine_size=1, castle_cnt=1)
    # graph = OctopusGraph(node_size=10, byzantine_size=1, castle_cnt=1, head_cnt=5,
    graph_class = getattr(graph, "CompleteGraph")
    graphs = graph_class(node_size=10, byzantine_size=1, castle_cnt=1, head_cnt=5,
                         head_byzantine_cnt=1, hand_byzantine_cnt=1, connected_p=0.7)
    graphs.show()


def test_wandb():
    # os.environ["WANDB_MODE"] = "offline"
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        }
    )

    # simulate training
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset

        # log metrics to wandb
        wandb.log({"acc": acc, "loss": loss})

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


# -*- coding: utf-8 -*-

def generate_directory_md(root_dir, indent=0, exclude_dirs=None):
    if exclude_dirs is None:
        exclude_dirs = []

    entries = os.listdir(root_dir)
    entries = [entry for entry in entries if entry not in exclude_dirs and not entry.startswith('.')]
    entries.sort()

    for entry in entries:
        full_path = os.path.join(root_dir, entry)
        if os.path.isdir(full_path):
            print("  " * indent + f"- **{entry}/**")
            generate_directory_md(full_path, indent + 1, exclude_dirs)
        else:
            print("  " * indent + f"- {entry}")


if __name__ == '__main__':
    # test_conf()
    main()
    # excluded_dirs = ["saved_models", "docs", "doc", "record",
    #                  "data", "logs", "wandb", "pictures", "reports",
    #                  "references", "notebooks"]
    #
    # print("# Directory Structure\n")
    # generate_directory_md(".", indent=0, exclude_dirs=excluded_dirs)
