import os
import random
import time
from os import path
import wandb
import numpy as np
import torch
from omegaconf import OmegaConf

import torchvision.models as models
from torchvision.models import ResNet18_Weights

from src.aggregations import aggregations
from src.attacks import attacks
from src.datas.federated_dataset import construct_federated_datasets
from src.library import graph
from src.library.cache_io import file_exist
from src.library.logger import create_logger
from src.library.tool import adapt_model_type
from src.tracking import metric

from src.train.nodes.base_node import BaseNode
from src.models.model import load_model
from src.train.controller.base_controller import BaseController

logger = create_logger()


class Coordinator(object):
    """Coordinator manages federated learning controller and nodes.
    A single instance of coordinator is initialized for each federated learning task
    when the package is imported.
    """

    def __init__(self):
        self.graph_ = None
        self.extra_info = None
        self.test_data_iter = None
        self.train_data_iter = None
        self.val_data_iter = None
        self.registered_model = False
        self.registered_dataset = False
        self.registered_controller = False
        self.registered_node = False
        self.registered_graph = False
        self.registered_attack = False
        self.registered_aggregation = False
        self.train_data = None
        self.test_data = None
        self.val_data = None
        self.conf = None
        self.model = None
        self._model_class = None
        self.controller = None
        self._controller_class = None
        self.nodes = None
        self._node_class = None
        self._agg_class = None
        self._attack_class = None
        self._graph_class = None
        self.agg_instance = None
        self.attack_instance = None

    def init(self, conf, init_all=True):
        """Initialize coordinator

        Args:
            conf (omegaconf.dictconfig.DictConfig): Internal configurations for federated learning.
            init_all (bool): Whether initialize dataset, model, controller, node, graph, aggregation and attack
                            other than configuration.
        """

        # For the online task, if online best model exist, we do not need to train again; if not, we train.
        if conf.task_name == metric.ONLINE_BEST_MODEL:
            best_title = metric.ONLINE_BEST_MODEL + ".pth"
            best_path_list = [metric.SAVED_MODELS, conf.data.dataset, conf.model,
                              metric.MEAN]
            if file_exist(best_title, best_path_list):
                logger.info("The best model exists.")
                return None
            if conf.node.calculate_static_regret is False:
                return None

        self.init_conf(conf)

        _set_random_seed(conf.seed)

        if init_all:
            self.init_dataset()

            self.init_model()

            self.init_graph()

            self.init_controller()

            self.init_nodes()

            self.init_aggregation()

            self.init_attack()

    def run(self):
        """Run the coordinator and the federated learning process.
        """
        start_time = time.time()

        if self.controller is not None:
            self.controller.start(self.model, self.nodes, self.graph_, self.agg_instance, self.attack_instance)
            logger.info("Total training time {:.1f}s".format(time.time() - start_time))
            print('-------------------------------------------')
            print("")

        # [optional] finish the wandb run, necessary in notebooks
        if self.conf is not None and self.conf.wandb_param.use_wandb:
            wandb.finish()

    def init_conf(self, conf):
        """Initialize coordinator configuration.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Configurations.
        """
        self.conf = conf
        if self.conf.gpu == -1 or not torch.cuda.is_available():
            self.conf.device = "cpu"
        else:
            self.conf.device = "cuda"

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.conf.gpu)

        # if we do not use momentum, we set momentum param in SGD as 1.
        if self.conf.node.optimizer.use_momentum is False:
            self.conf.node.momentum_controller = "ConstantLr"
            self.conf.lr_controller_param.init_momentum = 0
            self.conf.node.optimizer.use_another_momentum = False
            logger.warning("Because you do not use momentum, so we set momentum param in SGD as 1,"
                           " momentum controller as ConstantLr")

        logger.debug("Configurations: {}".format(self.conf))
        logger.debug("Device is {}{}.".format(self.conf.device, self.conf.gpu))

    def init_dataset(self):
        """Initialize datasets. Use provided datasets if not registered."""
        if self.registered_dataset:
            return
        self.train_data_iter, self.test_data_iter, self.extra_info = construct_federated_datasets(
            dataset_name=self.conf.data.dataset,
            num_of_par_nodes=self.conf.graph.nodes_cnt,
            partition_type=self.conf.data.partition_type,
            test_partition_type=self.conf.data.test_partition_type,
            class_per_node=self.conf.data.class_per_node,
            min_size=self.conf.data.min_size,
            alpha=self.conf.data.alpha,
            generator=self.conf.data.generator,
            train_batch_size=self.conf.data.train_batch_size,
            test_rate=self.conf.data.test_rate)
        self.conf.node.train_data_size_each = self.extra_info["train_data_size_each"]
        self.conf.node.test_data_size_each = self.extra_info["test_data_size_each"]
        logger.info("The dataset {} is inited, the data distribution is {}.".format(
            self.conf.data.dataset, self.conf.data.partition_type))

    def init_model(self):
        """Initialize model instance."""
        if not self.registered_model:
            self._model_class = load_model(self.conf.model)

        # model_class is None means model is registered as instance, no need initialization
        if self._model_class:
            self.model = self._model_class(feature_dimension=self.extra_info["feature_dimension"],
                                           num_classes=self.extra_info["num_classes"])
        self.adjust_model()

        # adapt model weight type with feature type
        self.model = adapt_model_type(self.model)

        logger.info("Model {} is inited.".format(self.conf.model))

    def adjust_model(self):
        """
        For the resnet18, we want to use the pretrained model, so we adjust model here.
        """
        # load pretrained model, only suit for model resnet18 and dataset Cifar10/Cifar100.
        if self.conf.model_pretrained and (self.conf.data.dataset == "Cifar10" or self.conf.data.dataset == "Cifar100") \
                and self.conf.model == "resnet18":
            # Load pre-trained weights from the torchvision model
            pretrained_resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            if self.conf.data.dataset == "Cifar10":
                # modify resnet18
                pretrained_resnet18.conv1 = torch.nn.Conv2d(pretrained_resnet18.conv1.in_channels,
                                                            pretrained_resnet18.conv1.out_channels,
                                                            3, 1, 1)
                pretrained_resnet18.maxpool = torch.nn.Identity()  # nn.Conv2d(64, 64, 1, 1, 1)
                num_features = pretrained_resnet18.fc.in_features
                pretrained_resnet18.fc = torch.nn.Linear(num_features, 10)

            self.model = pretrained_resnet18

    def init_controller(self):
        """Initialize a controller instance."""
        if not self.registered_controller:
            self._controller_class = BaseController
        kwargs = {}

        self.controller = self._controller_class(self.conf, **kwargs)

    def init_nodes(self):
        """Initialize node instances, each represent a federated learning node."""
        if not self.registered_node:
            self._node_class = BaseNode

        # Enforce system heterogeneity of nodes
        sleep_time = [0 for _ in range(self.conf.graph.nodes_cnt)]

        node_test_data = self.test_data_iter
        users = list(range(self.conf.graph.nodes_cnt))
        self.nodes = [self._node_class(u, self.conf,
                                       self.train_data_iter[u],
                                       node_test_data[u],
                                       self.conf.device,
                                       **{"sleep_time": sleep_time[i]})
                      for i, u in enumerate(users)]

        logger.info("Nodes in total: {}".format(len(self.nodes)))

    def init_node(self):
        """Initialize node instance.

        Returns:
            :obj:`BaseNode`: The initialized node instance.
        """
        if not self.registered_node:
            self._node_class = BaseNode

        # Get a random node if not specified
        if self.conf.index:
            user = list(range(self.conf.graph.nodes_cnt))[self.conf.index]
        else:
            user = random.choice(list(range(self.conf.graph.nodes_cnt)))

        return self._node_class(user,
                                self.conf,
                                self.train_data,
                                self.test_data,
                                self.conf.device)

    def init_graph(self):
        """
        Initialize the graph.
        """
        if not self.registered_graph:
            graph_class = getattr(graph, self.conf.graph.graph_type) if not self.conf.graph.centralized \
                else getattr(graph, "CompleteGraph")
        else:
            graph_class = self._graph_class

        if self.conf.controller.attack_type == "NoAttack" and self.conf.graph.byzantine_cnt != 0:
            self.conf.graph.byzantine_cnt = 0
            logger.info("Because the attack type is no attack, so we force to change the number of Byzantine "
                        "nodes in the graph to 0, otherwise it may affect model aggregation and training etc.")

        graph_ = graph_class(node_size=self.conf.graph.nodes_cnt, byzantine_size=self.conf.graph.byzantine_cnt,
                             centralized=self.conf.graph.centralized,
                             castle_cnt=self.conf.graph.castle_cnt, head_cnt=self.conf.graph.head_cnt,
                             head_byzantine_cnt=self.conf.graph.head_byzantine_cnt,
                             hand_byzantine_cnt=self.conf.graph.hand_byzantine_cnt,
                             connected_p=self.conf.graph.connectivity,
                             castle_k=self.conf.graph.castle_k,
                             conf=self.conf)
        # graph_.show()
        self.graph_ = graph_

        if self.graph_.node_size != self.conf.graph.nodes_cnt or self.graph_.byzantine_size != self.conf.graph.byzantine_cnt:
            logger.warning("Due to the graph type is not completed graph, "
                           "the graph node size or byzantine size maybe controlled by others graph parameters.")
            if self.graph_.node_size != self.conf.graph.nodes_cnt:
                logger.warning("So we change the configuration nodes_cnt from {} to the true graph nodes size {}.".
                               format(self.conf.graph.nodes_cnt, self.graph_.node_size))
                self.conf.graph.nodes_cnt = self.graph_.node_size
            if self.graph_.byzantine_size != self.conf.graph.byzantine_cnt:
                logger.warning("So we change the configuration byzantine_cnt from {} to the true graph byzantine "
                               "nodes size {}.".format(self.conf.graph.byzantine_cnt, self.graph_.byzantine_size))
                self.conf.graph.byzantine_cnt = self.graph_.byzantine_size

        network_type = "centralized" if self.conf.graph.centralized else "decentralized"
        logger.info("The network is {}. The graph is {}, has {} nodes, in which {} is honest, {} is byzantine.".
                    format(network_type, self.conf.graph.graph_type, self.graph_.node_size,
                           self.graph_.honest_size, self.graph_.byzantine_size))

    def init_aggregation(self):
        """
        Initialized the aggregation rule.
        """
        if self.registered_aggregation:
            agg_class = self._agg_class
        else:
            agg_class = getattr(aggregations, self.conf.controller.aggregation_rule)
        self.agg_instance = agg_class(graph=self.graph_,
                                      max_iter=self.conf.aggregation_param.max_iter,
                                      eps=self.conf.aggregation_param.eps,
                                      krmu_m=self.conf.aggregation_param.krum_m,
                                      exact_byz_cnt=self.conf.aggregation_param.exact_byz_cnt,
                                      byz_cnt=self.conf.aggregation_param.byz_cnt,
                                      weight_mh=self.conf.aggregation_param.weight_mh,
                                      threshold_selection=self.conf.aggregation_param.threshold_selection,
                                      threshold=self.conf.aggregation_param.threshold,
                                      conf=self.conf)
        logger.info("The aggregation rule is {}.".format(self.conf.controller.aggregation_rule))

    def init_attack(self):
        """
        Initialized the attack.
        """
        if self.registered_attack:
            attack_class = self._attack_class
        else:
            attack_class = getattr(attacks, self.conf.controller.attack_type)
        self.attack_instance = attack_class(graph=self.graph_,
                                            use_honest_mean=self.conf.attacks_param.use_honest_mean,
                                            mean=self.conf.attacks_param.mean,
                                            std=self.conf.attacks_param.std,
                                            sign_scale=self.conf.attacks_param.sign_scale,
                                            sample_scale=self.conf.attacks_param.sample_scale,
                                            little_scale=self.conf.attacks_param.little_scale,
                                            conf=self.conf)
        logger.info("The byzantine attack type is {}.".format(self.conf.controller.attack_type))

    def register_dataset(self, train_data_iter, test_data_iter, val_data_iter=None):
        """Register datasets.

        Datasets should inherit from :obj:`StackedTorchDataPackage`,
        and the train_data_iter should refer to federated_dataset.py.

        Args:
            train_data_iter (:obj:`DataGenerator`): Training dataset loader.
            test_data_iter (:obj:`DataGenerator`): Testing dataset loader.
            val_data_iter (:obj:`DataGenerator`): Validation dataset loader.
        """
        self.registered_dataset = True
        self.train_data_iter = train_data_iter
        self.test_data_iter = test_data_iter
        self.val_data_iter = val_data_iter

    def register_model(self, model):
        """Register customized model for federated learning.

        Args:
            model (nn.Module): PyTorch model, both class and instance are acceptable.
                Use model class when there is no specific arguments to initialize model.
        """
        self.registered_model = True
        if not isinstance(model, type):
            self.model = model
        else:
            self._model_class = model

    def register_controller(self, controller):
        """Register a customized federated learning controller.

        Args:
            controller (:obj:`BaseController`): Customized federated learning controller.
        """
        self.registered_controller = True
        self._controller_class = controller

    def register_node(self, node):
        """Register a customized federated learning node.

        Args:
            node (:obj:`BaseNode`): Customized federated learning node.
        """
        self.registered_node = True
        self._node_class = node

    def register_graph(self, graph):
        """Register a customized graph.

        Args:
            graph (:obj:`Graph`): Customized graph.
        """
        self.registered_graph = True
        self._graph_class = graph

    def register_aggregation(self, agg_class):
        """Register a customized aggregation class.

        Args:
            agg_class (:obj:`BaseAggregation`): Customized aggregation class.
        """
        self.registered_aggregation = True
        self._agg_class = agg_class

    def register_attack(self, attack_class):
        """Register a customized attack class.

        Args:
            attack_class (:obj:`BaseAggregation`): Customized attack class.
        """
        self.registered_attack = True
        self._agg_class = attack_class


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# Initialize the global coordinator object
_global_coord = Coordinator()


def init_conf(conf=None):
    """Initialize configuration for src. It overrides and supplements default configuration loaded from config.yaml
    with the provided configurations.

    Args:
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    here = path.abspath(path.dirname(__file__))
    config_file = path.join(here, 'config.yaml')
    return load_config(config_file, conf)


def load_config(file, conf=None):
    """Load and merge configuration from file and input

    Args:
        file (str): filename of the configuration.
        conf (dict): Configurations.

    Returns:
        omegaconf.dictconfig.DictConfig: Internal configurations managed by OmegaConf.
    """
    config = OmegaConf.load(file)
    if conf is not None:
        config = OmegaConf.merge(config, conf)
    return config


def init_logger(log_level):
    """
    Initialize internal logger.

    Args:
        log_level (int): Logger level, e.g., logging.INFO, logging.DEBUG

    """
    # TODO: the parameter in config only can control the log level in this python file,
    #  the other logger in other python files is default as logger.INFO
    global logger
    logger = create_logger(log_level=log_level)


def init_wandb(conf, config):
    """
    If you want to use wandb show your train process, set config.wandb_param.use_wandb be True.
    And you need do some pre-work as follows.
    1. Set up the wandb library:
            pip install wandb
    2. sign in: https://wandb.ai/site
    3. use your key to login in terminal:
            wandb login
    4. in main.py set config.wandb_param.use_wandb be True.
    """
    if config.wandb_param.use_wandb:
        if config.wandb_param.syn_to_web is False:
            os.environ["WANDB_MODE"] = "dryrun"
        is_centralized = "centralized" if config.graph.centralized else "decentralized"
        project_name = config.wandb_param.project_name if config.wandb_param.project_name != "" \
            else "my_{}_{}_{}_{}".format(config.task_name, config.data.dataset, config.model, is_centralized)

        wandb.init(project=project_name,
                   config=conf,
                   name="{}_{}_{}".format(config.controller.aggregation_rule, conf["controller"]["attack_type"],
                                          config.graph.graph_type[:4]))


def init(conf=None, init_all=True):
    """Initialize src.

    Args:
        conf (dict, optional): Configurations.
        init_all (bool, optional): Whether initialize dataset, model, controller, and node other than configuration.
    """
    global _global_coord

    config = init_conf(conf)

    init_logger(config.tracking.log_level)

    init_wandb(conf, config)

    _set_random_seed(config.seed)

    _global_coord.init(config, init_all)


def run():
    """Run federated learning process."""
    global _global_coord
    _global_coord.run()


def init_dataset():
    """Initialize dataset, either using registered dataset or out-of-the-box datasets set in config."""
    global _global_coord
    _global_coord.init_dataset()


def init_model():
    """Initialize model, either using registered model or out-of–the-box model set in config.

    Returns:
        nn.Module: Model used in federated learning.
    """
    global _global_coord
    _global_coord.init_model()

    return _global_coord.model


def get_coordinator():
    """Get the global coordinator instance.

    Returns:
        :obj:`Coordinator`: global coordinator instance.
    """
    return _global_coord


def register_dataset(train_data_iter, test_data_iter, val_data_iter=None):
    """Register datasets for federated learning training.

        Datasets should inherit from :obj:`StackedTorchDataPackage`,
        and the train_data_iter should refer to federated_dataset.py.

        Args:
            train_data_iter (:obj:`DataGenerator`): Training dataset loader.
            test_data_iter (:obj:`DataGenerator`): Testing dataset loader.
            val_data_iter (:obj:`DataGenerator`): Validation dataset loader.
    """
    global _global_coord
    _global_coord.register_dataset(train_data_iter, test_data_iter, val_data_iter)


def register_model(model):
    """Register model for federated learning training.

    Args:
        model (nn.Module): PyTorch model, both class and instance are acceptable.
    """
    global _global_coord
    _global_coord.register_model(model)


def register_controller(controller):
    """Register federated learning controller.

    Args:
        controller (:obj:`BaseController`): Customized federated learning controller.
    """
    global _global_coord
    _global_coord.register_controller(controller)


def register_node(node):
    """Register federated learning node.

    Args:
        node (:obj:`BaseNode`): Customized federated learning node.
    """
    global _global_coord
    _global_coord.register_node(node)


def register_graph(graph_class):
    """Register aggregation class.

    Args:
        graph_class (:obj:`Graph`): Customized graph class.
    """
    global _global_coord
    _global_coord.register_graph(graph_class)


def register_aggregation(agg_class):
    """Register aggregation class.

    Args:
        agg_class (:obj:`BaseAggregation`): Customized aggregation class.
    """
    global _global_coord
    _global_coord.register_aggregation(agg_class)


def register_attack(attack_class):
    """Register attack class.

    Args:
        attack_class (:obj:`BaseAttack`): Customized attack class.
    """
    global _global_coord
    _global_coord.register_attack(attack_class)
