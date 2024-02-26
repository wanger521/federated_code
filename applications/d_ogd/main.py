import argparse
import copy
import os
import sys

import wandb

sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])

import src
from src.library.logger import create_logger
from src.tracking import metric

REQUIRED_PYTHON = "python3"

logger = create_logger()


def run():
    parser = argparse.ArgumentParser(description='Online Application')
    parser.add_argument("--dataset", default="Mnist", choices=["Mnist", "Cifar10", "Cifar100", "Synthetic"],
                        help="dataset name", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--partition_type", type=str, default="iid", choices=["iid", "successive", "empty", "share",
                                                                              "noniid_class", "noniid_class_unbalance",
                                                                              "noniid_dir"],
                        help="Data partition type.")
    parser.add_argument("--test_rate", type=float, default=1,
                        help="the test using data rate, because the cpu memory constraint.")
    parser.add_argument("--model", choices=["rnn", "resnet", "resnet18", "resnet50",
                                            "vgg9", "simple_cnn", "softmax_regression",
                                            "linear_regression", "logistic_regression"],
                        default="softmax_regression", type=str)
    parser.add_argument("--gpu", type=int, default=0, help="default device No. of GPU, -1 means use cpu.")
    parser.add_argument("--epoch_or_iteration", type=str, default="iteration", choices=["epoch", "iteration"], help="")
    parser.add_argument("--rounds_iterations", type=int, default=5000, help="if iteration, then this means the total "
                                                                            "iterations.")
    parser.add_argument("--rounds", type=int, default=10, help="if epoch, then this means the total epochs.")
    parser.add_argument("--test_every_iteration", type=int, default=500, help="the interval of test.")
    parser.add_argument("--optimizer_type", type=str, default="SGD", choices=["SGD", "Adam"], help="")
    parser.add_argument("--weight_decay", type=float, default=0)
    # default False, not use_momentum
    parser.add_argument("--use_momentum", action="store_true",
                        help="SGD use momentum or not")
    parser.add_argument("--lr_controller", type=str, default="ConstantLr", choices=["ConstantLr", "OneOverSqrtKLr",
                                                                                    "OneOverKLr", "LadderLr",
                                                                                    "ConstantThenDecreasingLr",
                                                                                    "DecreasingStepLr",
                                                                                    "FollowOne"],
                        help="learning rate type.")
    parser.add_argument("--init_lr", type=float, default=0.1, help="")
    parser.add_argument("--init_momentum", type=float, default=0.1,
                        help="The momentum in SGD optimizer equal with 1-init_momentum")
    parser.add_argument("--graph_type", type=str, default="CompleteGraph", choices=["CompleteGraph", "ErdosRenyi",
                                                                                    "TwoCastle", "RingCastle",
                                                                                    "OctopusGraph"], help="")
    # default False, decentralized
    parser.add_argument("--centralized", action="store_true",
                        help="True means the frame work as centralized, else decentralized.")
    parser.add_argument("--nodes_cnt", type=int, default=10, help="The all nodes in graph.")
    parser.add_argument("--byzantine_cnt", type=int, default=2, help="The byzantine nodes in graph.")
    parser.add_argument("--aggregation_rule", type=str, default="Mean", choices=["Mean", "MeanWeightMH",
                                                                                 "NoCommunication", "Median",
                                                                                 "GeometricMedian", "Krum", "MKrum",
                                                                                 "TrimmedMean",
                                                                                 "RemoveOutliers", "Faba", "Phocas",
                                                                                 "IOS", "Brute", "Bulyan",
                                                                                 "CenteredClipping"], help="")
    parser.add_argument("--attack_type", type=str, default="NoAttack", choices=["NoAttack", "Gaussian", "SignFlipping",
                                                                                "SampleDuplicating", "ZeroValue",
                                                                                "Isolation", "LittleEnough"], help="")
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument("--use_honest_mean", type=int, default=1, choices=[0, 1],
                        help="0 means attack does not use honest mean, 1 means use honest mean")

    args = parser.parse_args()
    logger.info(f"arguments: {args}")

    config = {
        "data": {"dataset": args.dataset, "partition_type": args.partition_type,
                 "train_batch_size": args.batch_size, "test_rate": args.test_rate},
        "model": args.model,
        "gpu": args.gpu,
        "node": {"epoch_or_iteration": args.epoch_or_iteration,
                 "optimizer": {"type": args.optimizer_type, "use_momentum": args.use_momentum,
                               "use_another_momentum": True, "weight_decay": args.weight_decay},
                 "lr_controller": args.lr_controller, "batch_size": args.batch_size,
                 "message_type_of_node_sent": "model", "local_epoch": 1,
                 "local_iteration": 1, "momentum_controller": "FollowOne"
                 },
        "graph": {"centralized": args.centralized, "graph_type": args.graph_type,
                  "byzantine_cnt": args.byzantine_cnt, "nodes_cnt": args.nodes_cnt},
        "controller": {"nodes_per_round": args.nodes_cnt, "random_selection": False,
                       "aggregation_rule": args.aggregation_rule, "attack_type": args.attack_type,
                       "rounds_iterations": args.rounds_iterations, "rounds": args.rounds,
                       "test_every_iteration": args.test_every_iteration},
        "lr_controller_param": {"init_lr": args.init_lr, "init_momentum": args.init_momentum},
        "aggregation_param": {"exact_byz_cnt": True, "byz_cnt": 0, "weight_mh": True,
                              "threshold_selection": "true", "threshold": 0.1},
        "attacks_param": {"use_honest_mean": bool(args.use_honest_mean), "sign_scale": -4, "little_scale": None,
                          "std": 1},
        "wandb_param": {"use_wandb": True, "project_name": "", "syn_to_web": True}
    }

    # next two line just for th graduate experiment
    if config["data"]["partition_type"] == "noniid_class_unbalance":
        if config["data"]["dataset"] == "Mnist":
            config["data"]["class_per_node"] = 1
        elif config["data"]["dataset"] == "Cifar10":
            config["data"]["class_per_node"] = 4

    if "node" in config:
        config["node"]["calculate_static_regret"] = True
    else:
        config["node"] = {"calculate_static_regret": True}
    config["controller"]["print_interval"] = config["controller"]["test_every_iteration"]

    config_online = copy.deepcopy(config)
    config_online["task_name"] = metric.ONLINE_BEST_MODEL
    config_online["data"]["partition_type"] = "iid"
    config_online["controller"] = {"nodes_per_round": 1, "random_selection": False,
                                   "aggregation_rule": "Mean", "attack_type": "NoAttack"}
    if "rounds" in config["controller"]:
        config_online["controller"]["rounds"] = 10 * config["controller"]["rounds"]
    else:
        config_online["controller"]["rounds"] = 100
    if "rounds_iterations" in config["controller"]:
        config_online["controller"]["rounds_iterations"] = 10 * config["controller"]["rounds_iterations"]
    else:
        config_online["controller"]["rounds_iterations"] = 50000

    # config_online["node"]["batch_size"] = config["node"]["batch_size"] * config["graph"]["nodes_cnt"]
    config_online["lr_controller_param"]["boundary_epoch"] = int(config_online["controller"]["rounds"] * 0.3)
    config_online["lr_controller_param"]["boundary_iteration"] = int(
        config_online["controller"]["rounds_iterations"] * 0.3)
    config_online["controller"]["save_model_every_epoch"] = config_online["controller"]["rounds"]
    config_online["controller"]["save_model_every_iteration"] = config_online["controller"]["rounds_iterations"]
    config_online["graph"] = {"centralized": True, "nodes_cnt": 1, "byzantine_cnt": 0}
    config_online["wandb_param"]["use_wandb"] = False
    config_online["node"]["optimizer"]["use_momentum"] = True
    if args.dataset == "Mnist":
        config_online["lr_controller_param"]["init_lr"] = 0.1
        config_online["node"]["lr_controller"] = "ConstantLr"

    if config["data"]["dataset"] == "Cifar100":
        config_online["lr_controller_param"]["init_lr"] = config["lr_controller_param"]["init_lr"] \
                                                          * 3
        config_online["data"]["train_batch_size"] = config["data"]["train_batch_size"] * 4


    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #  If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=256"

    # wandb.init(project="my-online-mnist-project",
    #            config=config)

    src.init(config_online)
    src.run()

    src.init(config)
    src.run()

    # # [optional] finish the wandb run, necessary in notebooks
    # wandb.finish()


if __name__ == '__main__':
    run()
