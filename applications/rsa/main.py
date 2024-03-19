import argparse
import sys, os

sys.path.append(os.path.split(os.path.split(sys.path[0])[0])[0])
from applications.rsa.rsa_controller import RSAController
import src
from applications.byrd_saga.byrd_saga_node import ByrdSagaNode
from src.coordinate import init_conf, init_logger
from src.library.logger import create_logger
from src.tracking import metric

REQUIRED_PYTHON = "python3"

logger = create_logger()


def run():
    parser = argparse.ArgumentParser(description='RSA Application')
    parser.add_argument("--dataset", default="Mnist", choices=["Mnist", "Cifar10", "Cifar100", "Synthetic",
                                                               "FashionMnist", "ImageNet"],
                        help="dataset name", type=str)
    parser.add_argument("--batch_size", default=32, type=int)  # we suggest small batch_size for byrd_saga.
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
    parser.add_argument("--rounds_iterations", type=int, default=2000, help="if iteration, then this means the total "
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
    parser.add_argument("--init_lr", type=float, default=0.1, help="For Synthetic data, recommend init_lr be small, "
                                                                   "like 0.00001.")
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
                                                                                 "CenteredClipping", "SignGuard",
                                                                                 "Dnc"],
                        help="")
    parser.add_argument("--attack_type", type=str, default="NoAttack", choices=["NoAttack", "Gaussian", "SignFlipping",
                                                                                "SampleDuplicating", "ZeroValue",
                                                                                "Isolation", "LittleEnough", "AGRFang",
                                                                                "AGRTailored"], help="")
    parser.add_argument("--seed", type=int, default=0, help="")
    # The RSA performance is sensitive to this lambda_rsa, small lambda_rsa is preferred.
    parser.add_argument("--lambda_rsa", type=float, default=0.001, help="the parameter of rsa")
    parser.add_argument("--paradigm_num", type=int, default=1, help="the parameter of rsa, only achieve 1 paradigm.")
    parser.add_argument("--weight_decay_rsa", type=float, default=0.01, help="the weight decay of rsa regularisation "
                                                                             "term.")

    args = parser.parse_args()
    logger.info(f"arguments: {args}")

    config = {
        "seed": args.seed,
        "task_name": "RSA",
        "data": {"dataset": args.dataset, "partition_type": args.partition_type,
                 "train_batch_size": args.batch_size, "test_rate": args.test_rate,
                 "generator": "random"},
        "model": args.model,
        "gpu": args.gpu,
        "node": {"epoch_or_iteration": args.epoch_or_iteration,
                 "optimizer": {"type": args.optimizer_type, "use_momentum": args.use_momentum,
                               "use_another_momentum": False, "weight_decay": args.weight_decay},
                 "lr_controller": args.lr_controller, "batch_size": args.batch_size,
                 "message_type_of_node_sent": "gradient", "local_epoch": 1,
                 "local_iteration": 1, "momentum_controller": "FollowOne",
                 },
        "graph": {"centralized": args.centralized, "graph_type": args.graph_type,
                  "byzantine_cnt": args.byzantine_cnt, "nodes_cnt": args.nodes_cnt},
        "controller": {"nodes_per_round": args.nodes_cnt, "random_selection": False,
                       "aggregation_rule": args.aggregation_rule, "attack_type": args.attack_type,
                       "rounds_iterations": args.rounds_iterations, "rounds": args.rounds,
                       "test_every_iteration": args.test_every_iteration},
        "lr_controller_param": {"init_lr": args.init_lr, "init_momentum": args.init_momentum,
                                "a": 1, "b": 1},
        "aggregation_param": {"exact_byz_cnt": True, "byz_cnt": 0, "weight_mh": True,
                              "threshold_selection": "parameter", "threshold": 0.1},
        "attacks_param": {"use_honest_mean": True, "sign_scale": -4, "little_scale": None},
        "wandb_param": {"use_wandb": False, "project_name": "", "syn_to_web": True},
        "rsa_param": {"lambda_rsa": args.lambda_rsa, "paradigm_num": args.paradigm_num,
                      "weight_decay_rsa": args.weight_decay_rsa}
    }

    # We do not successfully achieve byrd_saga + momentum, so we force to set use_momentum = fasle.
    config["node"]["optimizer"]["use_momentum"] = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    #  If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb=256"

    # register the  RSAController as Controller Class.
    src.register_controller(RSAController)

    # replace the config.yaml in rsa dictionary to replace in federated.
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
    config = src.load_config(config_file, config)

    src.init(config)
    src.run()


if __name__ == '__main__':
    run()
