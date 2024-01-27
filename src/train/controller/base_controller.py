import copy
import os
import time
import wandb

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from src.library.cache_io import dump_file_in_cache, dump_file_in_root, file_exist, dump_model_in_root
from src.tracking import metric
from src.library.float import rounding
from src.library.logger import create_logger

logger = create_logger()


class BaseController(object):
    """Default implementation of federated learning controller.

    Args:
        conf (omegaconf.dictconfig.DictConfig): Configurations of EasyFL.
        test_data (:obj:`FederatedDataset`): Test dataset for centralized testing in controller, optional.
        val_data (:obj:`FederatedDataset`): Validation dataset for centralized validation in controller, optional.

    Override the class and functions to implement customized controller.

    Example:
        >>> from src.train.controller.base_controller import BaseController
        >>> class CustomizedController(BaseController):
        >>>     def __init__(self, conf, test_data=None, val_data=None):
        >>>         super(CustomizedController, self).__init__(conf, test_data, val_data)
        >>>         pass  # more initialization of attributes.
        >>>
        >>>     def aggregate_and_attack(self):
        >>>         # Implement customized aggregate_and_attack method, which overwrites the default method.
        >>>         pass
    """

    def __init__(self,
                 conf,
                 test_data=None,
                 val_data=None):
        self.title = None
        self.graph_ = None
        self.conf = conf
        self.test_data = test_data
        self.val_data = val_data
        self._is_training = False
        self._should_stop = False

        self._current_round = -1
        self._node_uploads = {}
        self.controller_retain_one_model = self.if_controller_retain_one_model()
        self._model = [None] if self.controller_retain_one_model is True else [None for _ in
                                                                               range(self.conf.graph.nodes_cnt)]
        self._compressed_model = [None] if self.controller_retain_one_model is True \
            else [None for _ in range(self.conf.graph.nodes_cnt)]
        self.controller_avg_model = None
        self.messages = None
        self._nodes = None
        self._etcd = None
        self.selected_nodes = []
        self.agg_class = None
        self.attack_class = None

        self._controller_metric = None
        self._round_time = None
        self._begin_train_time = None  # training begin time for a round
        self._start_time = None  # training start time for a task
        self.node_stubs = {}

        self._cumulative_times = []  # cumulative training after each test
        self._train_accuracies = {}
        self._train_losses = {}
        self._train_regrets = {}
        self._train_node_num = []
        self._test_accuracies = {}
        self._test_losses = {}
        self._test_consensus_errors = []
        self._test_avg_model_accuracies = {}
        self._test_avg_model_losses = {}
        self._test_node_num = []
        self._cumulative_test_round = []  # record the corresponding test round

        self.wandb_message = {}

    def start(self, model, nodes, graph_, agg_class, attack_class):
        """Start federated learning process, including training and testing.

        Args:
            model (nn.Module): The model to train.
            nodes (list[:obj:`BaseNode`]|list[str]): Available nodes.
                Nodes are actually node grpc addresses when in remote training.
            graph_: The graph of network.
            agg_class: The aggregation rule object.
            attack_class: The attack type.
        """
        # Setup
        self._start_time = time.time()
        self._reset()
        list_model = self.get_all_node_list_model(model)
        self.set_model(list_model)
        self.set_nodes(nodes)
        self.graph_ = graph_
        self.agg_class = agg_class
        self.attack_class = attack_class

        self.init_train_test_track()
        self.init_title()

        while not self.should_stop():
            self._round_time = time.time()

            self._current_round += 1
            self.print_("\n-------- round {} --------".format(self._current_round))

            # if self._current_round == 0, we do not train, only record train accuracy loss regret as 0,
            # we already do this in init_train_test_track.
            if self._current_round != 0:
                # Train
                self.pre_train()
                self.train()
                self.post_train()
            else:
                self.selected_nodes = self._nodes

            # Test
            test_round = self.conf.controller.rounds if self.conf.node.epoch_or_iteration == "epoch" else \
                self.conf.controller.rounds_iterations
            if self._do_every(self.conf.controller.test_every_epoch, self.conf.controller.test_every_iteration,
                              self._current_round, test_round):
                self.pre_test()
                self.test()
                self.post_test()

            # Save Model
            self.save_model()

        logger.info("Consensus errors: {}".format(rounding(self._test_consensus_errors, 8)))
        logger.info("Accuracies: {}".format(rounding(self._test_accuracies["avg"], 4)))
        logger.info("Losses: {}".format(rounding(self._test_losses["avg"], 6)))
        logger.info("Cumulative training time: {}".format(rounding(self._cumulative_times, 2)))
        self.save_train_test_data()

    def stop(self):
        """Set the flag to indicate training should stop."""
        self._should_stop = True

    def pre_train(self):
        """Preprocessing before training."""
        pass

    def if_controller_retain_one_model(self):
        """The controller whether retain one model, if the graph is centralized, and the nodes
        exchange gradients and do not do local model iterations , or nodes exchange models,
        the controller only need to retain one consistent model, and aggregate once.
        Else, it retains a node size model list.

         Return:
             bool: True, False

         Note:
             do not do local model iterations means  conf.node.epoch_or_iteration=="iteration" and "
             conf.node.local_iteration==1.
         """
        controller_retain_one_model = False
        if self.conf.graph.centralized is True:
            if self.conf.node.epoch_or_iteration == "iteration" and self.conf.node.local_iteration == 1:
                controller_retain_one_model = True
            elif self.conf.node.message_type_of_node_sent == "model":
                controller_retain_one_model = True
            else:
                logger.warning("This experiment is set up as centralized, requiring local updates and communicating"
                               " gradient information. The local update setting leads to inconsistencies in the local "
                               "models of the nodes, and although the graph is centralized, this code goes about"
                               "running this scenario in a decentralized manner, i.e., it allows for different node "
                               "local models to be held between different nodes, and the code implementation "
                               "is also implemented essentially in a decentralized architecture.")
                logger.warning("Note that this implementation is not a common implementation; "
                               "typically the case where there is a centre containing local updates will swap that"
                               " variable of the model, but this code does not implement this approach.")
                logger.warning("中文解释：本实验设置为有中心的，需要本地更新，且交流梯度信息。本地更新设置会导致节点的本地模型不一致，"
                               "虽然图为有中心的，但本代码以无中心的方式去运行这种情况，即允许不同节点之间持有不同的本地模型，"
                               "在代码实现上也基本以无中心的架构实现。注意，这种实现并不是常见的实现方式，"
                               "一般有中心包含本地更新的情况下，会交换模型的改变量，但是本代码没有实现这种方法。")
        return controller_retain_one_model

    def get_all_node_list_model(self, model):
        """
        Reset the coordinate sent model, if we only need to retain one centralized model, means
         controller_retain_one_model = True, we let list_model len is 1, else we set is self.conf.graph.nodes_cnt.
        """
        if self.controller_retain_one_model is True:
            list_model = [model.to(self.conf.device)]
        else:
            list_model = [model.to(self.conf.device) for _ in range(self.conf.graph.nodes_cnt)]
        return list_model

    def train(self):
        """Training process of federated learning."""
        self.print_("--- start training ---")

        self.selection(self._nodes, self.conf.controller.nodes_per_round)
        self.compression()

        begin_train_time = time.time()

        self.distribution_to_train()
        self.aggregate_and_attack()

        train_time = time.time() - begin_train_time
        self.track_train_results(train_time)

    def post_train(self):
        """Postprocessing after training."""
        pass

    def pre_test(self):
        """Preprocessing before testing."""
        pass

    def test(self):
        """Testing process of federated learning."""
        self.print_("--- start testing ---")

        test_begin_time = time.time()

        self.test_in_node()

        test_time = time.time() - test_begin_time
        self.track_test_results(test_time)

    def post_test(self):
        """Postprocessing after testing."""
        pass

    def should_stop(self):
        """Check whether should stop training. Stops the training under two conditions:
        1. Reach max number of training rounds
        2. TODO: Accuracy higher than certain amount.

        Returns:
            bool: A flag to indicate whether should stop training.
        """
        if self.conf.node.epoch_or_iteration == "iteration":
            if self._should_stop or (
                    self.conf.controller.rounds_iterations and self._current_round >=
                    self.conf.controller.rounds_iterations):
                self._is_training = False
                return True
        else:
            if self._should_stop or (
                    self.conf.controller.rounds and self._current_round >= self.conf.controller.rounds):
                self._is_training = False
                return True
        return False

    def test_in_node(self):
        """Conduct testing in nodes.
        Currently, it supports testing on the selected nodes for training.

        """
        self.compression()
        self.distribution_to_test()

    def selection(self, nodes, nodes_per_round):
        """Select a fraction of total nodes for training.
        Two selection strategies are implemented: 1. random selection; 2. select the first K nodes.

        Args:
            nodes (list[:obj:`BaseNode`]|list[str]): Available nodes.
            nodes_per_round (int): Number of nodes to participate in training each round.

        Returns:
            (list[:obj:`BaseNode`]|list[str]): The selected nodes.
        """
        if nodes_per_round > len(nodes) and self._current_round == 0:
            logger.warning("Available nodes for selection are smaller than required nodes for each round")

        nodes_per_round = min(nodes_per_round, len(nodes))
        if self.conf.controller.random_selection:
            np.random.seed(self._current_round * 2)
            self.selected_nodes = np.random.choice(nodes, nodes_per_round, replace=False)
        else:
            self.selected_nodes = nodes[:nodes_per_round]

        return self.selected_nodes

    def compression(self):
        """Model compression to reduce communication cost."""
        self._compressed_model = self._model

    def distribution_to_train(self):
        """Distribute model and configurations to selected nodes to train."""
        if self.controller_retain_one_model is True:
            self.distribution_to_train_centralized()
        else:
            self.distribution_to_train_decentralized()

    def distribution_to_train_centralized(self):
        """Conduct training sequentially for selected nodes in the group. The graph is centralized."""
        uploaded_models = {}
        uploaded_gradient = {}  # for conf.node. message_type_of_node_sent is gradient
        uploaded_weights = {}
        uploaded_metrics = {}
        for node in self.selected_nodes:
            # Update node config before training
            self.conf.node.task_id = self.conf.task_id
            self.conf.node.round_id = self._current_round
            self.conf.node.model = self.conf.model

            uploaded_content = node.run_train(self._compressed_model[0], self.conf)

            model = self.decompression(uploaded_content[metric.MODEL])
            uploaded_models[node.cid] = model
            uploaded_gradient[node.cid] = uploaded_content[metric.GRADIENT_MESSAGE]
            uploaded_weights[node.cid] = uploaded_content[metric.TRAIN_DATA_SIZE]
            uploaded_metrics[node.cid] = uploaded_content[metric.METRIC]

        self.set_node_uploads_train(uploaded_models, uploaded_weights, uploaded_gradient, uploaded_metrics)

    def distribution_to_train_decentralized(self):
        """Conduct training sequentially for selected nodes in the group. The graph is decentralized."""
        uploaded_models = {}
        uploaded_gradient = {}  # for conf.node. message_type_of_node_sent is gradient
        uploaded_weights = {}
        uploaded_metrics = {}
        for node in self.selected_nodes:
            # Update node config before training
            self.conf.node.task_id = self.conf.task_id
            self.conf.node.round_id = self._current_round
            self.conf.node.model = self.conf.model

            uploaded_content = node.run_train(self._compressed_model[node.cid], self.conf)

            model = self.decompression(uploaded_content[metric.MODEL])
            uploaded_models[node.cid] = model
            uploaded_gradient[node.cid] = uploaded_content[metric.GRADIENT_MESSAGE]
            uploaded_weights[node.cid] = uploaded_content[metric.TRAIN_DATA_SIZE]
            uploaded_metrics[node.cid] = uploaded_content[metric.METRIC]

        self.set_node_uploads_train(uploaded_models, uploaded_weights, uploaded_gradient, uploaded_metrics)

    def distribution_to_test(self):
        """Distribute to conduct testing on nodes."""
        if self.controller_retain_one_model is True:
            self.distribution_to_test_centralized()
        else:
            self.distribution_to_test_decentralized()

    def distribution_to_test_centralized(self):
        """Conduct testing sequentially for selected testing nodes in centralized."""
        uploaded_accuracies = {}
        uploaded_losses = {}
        uploaded_data_sizes = {}
        uploaded_consensus_error = 0

        test_nodes = self.get_test_nodes()
        for node in test_nodes:
            # Update node config before testing
            self.conf.node.task_id = self.conf.task_id
            self.conf.node.round_id = self._current_round

            uploaded_content = node.run_test(self._compressed_model[0], self.conf)

            uploaded_accuracies[node.cid] = uploaded_content[metric.TEST_ACCURACY]
            uploaded_losses[node.cid] = uploaded_content[metric.TEST_LOSS]
            uploaded_data_sizes[node.cid] = uploaded_content[metric.TEST_DATA_SIZE]

        self.set_node_uploads_test(uploaded_accuracies, uploaded_losses, uploaded_consensus_error,
                                   uploaded_data_sizes)
        self.set_node_avg_model_uploads_test(uploaded_accuracies, uploaded_losses)

    def distribution_to_test_decentralized(self):
        """Conduct testing sequentially for selected testing nodes in decentralized."""
        uploaded_accuracies = {}
        uploaded_losses = {}
        uploaded_data_sizes = {}
        uploaded_avg_model_accuracies = {}
        uploaded_avg_model_losses = {}
        uploaded_avg_model_data_sizes = {}
        uploaded_consensus_error = self.test_measure_consensus_error()

        test_nodes = self.get_test_nodes()
        controller_avg_model = self.get_controller_avg_model()

        for node in test_nodes:
            # Update node config before testing
            self.conf.node.task_id = self.conf.task_id
            self.conf.node.round_id = self._current_round

            uploaded_content = node.run_test(self._compressed_model[node.cid], self.conf)
            uploaded_accuracies[node.cid] = uploaded_content[metric.TEST_ACCURACY]
            uploaded_losses[node.cid] = uploaded_content[metric.TEST_LOSS]
            uploaded_data_sizes[node.cid] = uploaded_content[metric.TEST_DATA_SIZE]

            uploaded_content = node.run_avg_model_test(controller_avg_model, self.conf)
            uploaded_avg_model_accuracies[node.cid] = uploaded_content[metric.TEST_ACCURACY]
            uploaded_avg_model_losses[node.cid] = uploaded_content[metric.TEST_LOSS]
            uploaded_avg_model_data_sizes[node.cid] = uploaded_content[metric.TEST_DATA_SIZE]

        self.set_node_uploads_test(uploaded_accuracies, uploaded_losses, uploaded_consensus_error,
                                   uploaded_data_sizes)
        self.set_node_avg_model_uploads_test(uploaded_avg_model_accuracies, uploaded_avg_model_losses)
        # torch.cuda.empty_cache()

    @torch.no_grad()
    def get_controller_avg_model(self):
        """
        Because decentralized learning want know the average model of all honest node model the performance,
        so we do another test on this controller average model.
        """
        model_after_agg = self.get_model()
        if self.controller_retain_one_model:
            controller_avg_model = model_after_agg[0]
        else:
            controller_avg_model = copy.deepcopy(model_after_agg[0])
            avg_messages = self.models_avg(model_after_agg, self.get_test_nodes())
            cumulated_param = 0
            for _, param in controller_avg_model.named_parameters():
                param_size = param.nelement()
                beg, end = cumulated_param, cumulated_param + param_size
                param.data = avg_messages[beg:end].view_as(param)
                cumulated_param = end
        return controller_avg_model

    @torch.no_grad()
    def models_avg(self, models, nodes):
        """
        Get the honest model average message.
        """
        messages = []
        nodes_cid = []
        for node in nodes:
            if node.cid in self.graph_.honest_nodes:
                nodes_cid.append(node.cid)
        for node_cid, model in enumerate(models):
            if node_cid in nodes_cid:
                model_param = parameters_to_vector([param.data for param in model.parameters()])
                messages.append(model_param)
        messages = torch.stack(messages, dim=0)
        avg_messages = torch.mean(messages, dim=0)
        return avg_messages

    @torch.no_grad()
    def test_measure_consensus_error(self):
        """
        In decentralized setting (means the nodes local models are different),
        we extra measure the consensus error of test nodes' models .
        """
        test_nodes = self.get_test_nodes()
        test_nodes_cid = [node.cid for node in test_nodes]

        messages = []
        models = self.get_model()
        for node_cid, model in enumerate(models):
            if node_cid not in self.graph_.honest_nodes:
                continue
            model_param = parameters_to_vector([param.data for param in model.parameters()])
            messages.append(model_param)
        messages = torch.stack(messages, dim=0)
        return torch.var(messages,
                         dim=0, unbiased=False).norm().item()

    def get_test_nodes(self):
        """Get nodes to run testing.

        Returns:
            (list[:obj:`BaseNode`]|list[str]): Nodes to test.
        """
        if self.conf.controller.test_all:
            test_nodes = self._nodes
        else:
            # For the initial testing, if no nodes are selected, test all nodes
            test_nodes = self.selected_nodes if self.selected_nodes is not None else self._nodes
            if not self.conf.controller.test_byzantine:
                test_nodes = [node for node in test_nodes if node.cid in self.graph_.honest_nodes]
        return test_nodes

    def _mean_value(self, values):
        return np.mean(values)

    def _weighted_value(self, values, weights):
        return np.average(values, weights=weights)

    def decompression(self, model):
        """Decompression the models from nodes"""
        return model

    def aggregate_and_attack(self):
        """Byzantine nodes send arbitrary adversary messages,
        controller aggregate these training messages updates from nodes.
        Controller aggregates trained models or gradients from nodes via aggregation rule.
        """
        # get the self._models and base the upload model to update models
        uploaded_content = self.get_node_uploads()
        if self.controller_retain_one_model:
            models = self.get_model() * self.graph_.node_size
        else:
            models = self.get_model()

        for node in self.selected_nodes:
            models[node.cid] = uploaded_content[metric.MODEL][node.cid]

        selected_nodes_cid = [node.cid for node in self.selected_nodes]

        # if no nodes need to aggregate, jump the aggregate and attack step.
        if len(selected_nodes_cid) == 0:
            return None

        # get the model parameter or gradient as a nodes_cnt dimension tensor vector.
        # The model or gradient for one node is combined as one dimension tensor vector.
        messages = []
        for node_cid, model in enumerate(models):
            if self.conf.node.message_type_of_node_sent == "model":
                model_param = parameters_to_vector([param.data for param in model.parameters()])
                messages.append(model_param)
            else:
                gradient_param = uploaded_content[metric.GRADIENT_MESSAGE][node_cid]
                messages.append(gradient_param)
        messages = torch.stack(messages, dim=0)
        self.messages = messages
        new_messages = [None for _ in range(self.graph_.node_size)]#copy.deepcopy(messages)

        # attack and aggregate
        for node in selected_nodes_cid:
            if not self.conf.controller.byzantine_actually_train and node not in self.graph_.honest_nodes:
                continue
            # attack
            base_messages = self.attack(selected_nodes_cid=selected_nodes_cid, node=node, new_graph=None)
            # aggregate
            new_messages[node] = self.aggregate(all_messages=base_messages, selected_nodes_cid=selected_nodes_cid,
                                                node=node, new_graph=None)
            # centralized only need to do attack and aggregate once.
            if self.controller_retain_one_model:
                new_messages[0] = new_messages[node]
                models = models[: 1]
                break

        # Base the aggregation results, update the nodes model
        flag = 0
        for node in self.selected_nodes:
            node_cid = node.cid
            if self.controller_retain_one_model:
                if flag != 0:
                    break
                else:
                    node_cid = 0
                    flag = 1
            new_message = new_messages[node_cid] if new_messages[node_cid] is not None else self.messages[node_cid]
            self.one_step_gradient_descent_update_or_parameter_update(
                models[node_cid], new_message,
                uploaded_content[metric.NODE_METRICS][node_cid][metric.LEARNING_RATE])
        self.set_model(models, load_dict=True)

    def attack(self, selected_nodes_cid, node, new_graph=None):
        """Byzantine nodes change the messages which sent to the others.
        Args:
            selected_nodes_cid (list[int]): List of select node cid list.
            node (int): the aggregate node cid.
            new_graph (graph): If you want to use the time-varying graph, you can extend with this variable.
        Returns:
            torch.Tensor: Attack messages, vector in nodes_cnt dimension.
        """
        return self.attack_class.run(all_messages=self.messages, selected_nodes_cid=selected_nodes_cid,
                                     node=node, new_graph=new_graph)

    def aggregate(self, all_messages, selected_nodes_cid, node, new_graph=None):
        """Aggregate models uploaded from nodes via aggregation rule.

        Args:
            all_messages (tensor): aggregate messages.
            selected_nodes_cid (list[int]): List of select node cid list.
            node (int): the aggregate node cid.
            new_graph (graph): If you want to use the time-varying graph, you can extend with this variable.
        Returns:
            torch.Tensor: Aggregated message, vector in one dimension.
        """
        return self.agg_class.run(all_messages=all_messages, selected_nodes_cid=selected_nodes_cid,
                                  node=node, new_graph=new_graph)[0]

    def one_step_gradient_descent_update_or_parameter_update(self, model, new_message, lr):
        """
        This method is supported for controller. Because controller has to aggregate model or gradient,
        so the node offer this method to based on the information go and update the model parameters manually
        or do a step of gradient descent.
        """
        cumulated_param = 0
        for _, param in model.named_parameters():
            param_size = param.nelement()
            beg, end = cumulated_param, cumulated_param + param_size
            if self.conf.node.message_type_of_node_sent == "model":
                param.data = new_message[beg:end].view_as(param)
            elif self.conf.node.message_type_of_node_sent == "gradient":
                param.data.add_(new_message[beg:end].view_as(param), alpha=-lr)
            else:
                raise ValueError('invalid conf.node.message_type_of_node_sent type.')
            cumulated_param = end

    def _reset(self):
        self._current_round = -1
        self._should_stop = False
        self._is_training = True

    def is_training(self):
        """Check whether the controller is in training or has stopped training.

        Returns:
            bool: A flag to indicate whether controller is in training.
        """
        return self._is_training

    def set_model(self, model, load_dict=False):
        """Update the universal model in the controller.

        Args:
            model (nn.Module): New model for self.controller_retain_one_model is True,
                                new model list for self.controller_retain_one_model is False.
            load_dict (bool): A flag to indicate whether load state dict or copy the model.
        """
        if self.controller_retain_one_model is True:
            if load_dict and self._model[0] is not None:
                self._model[0].load_state_dict(model[0].state_dict())
            else:
                self._model[0] = copy.deepcopy(model[0])
        else:
            if load_dict and None not in self._model:
                for i in range(len(model)):
                    self._model[i].load_state_dict(model[i].state_dict())
            else:
                for i in range(len(model)):
                    self._model[i] = copy.deepcopy(model[i])

    def get_model(self):
        """Get self._model."""
        return self._model

    def set_nodes(self, nodes):
        self._nodes = nodes

    def num_of_nodes(self):
        return len(self._nodes)

    def reset_graph_time_varying(self):
        # TODO
        pass

    def save_model(self):
        """Save the model in the controller."""
        save_time = self.conf.controller.rounds_iterations if self.conf.node.epoch_or_iteration == "iteration" \
            else self.conf.controller.rounds
        if self._do_every(self.conf.controller.save_model_every_epoch, self.conf.controller.save_model_every_iteration,
                          self._current_round, save_time) and self._current_round != 0:
            # and self.is_primary_controller():
            save_path = self.conf.controller.save_model_path
            if self.conf.task_name == metric.ONLINE_BEST_MODEL:
                title = metric.ONLINE_BEST_MODEL + ".pth"
            else:
                title = "{}_r_{}.pth".format(self.title, self._current_round)
            path_list = save_path
            if save_path == "":
                path_list = [metric.SAVED_MODELS, self.conf.data.dataset, self.conf.model,
                             self.conf.controller.aggregation_rule]
                save_path = os.path.join(os.getcwd(), *path_list)
            save_path = os.path.join(save_path, title)
            if self.controller_retain_one_model is True:
                dump_model_in_root(title, self._model[0].cpu().state_dict(), path_list)
            else:
                dump_model_in_root(title, [self._model[i].cpu().state_dict() for i in range(len(self._model))],
                                   path_list)
            logger.info("Model saved at {}".format(save_path))

    def set_node_uploads_train(self, models, weights, gradient, metrics=None):
        """Set training updates uploaded from nodes.

        Args:
            models (dict): A collection of models.
            weights (dict): A collection of weights.
            gradient (dict): A collection of gradient.
            metrics (dict): Node training metrics.
        """
        self.set_node_uploads(metric.MODEL, models)
        self.set_node_uploads(metric.TRAIN_DATA_SIZE, weights)
        self.set_node_uploads(metric.GRADIENT_MESSAGE, gradient)
        self.set_node_uploads(metric.NODE_METRICS, metrics)

    def set_node_uploads_test(self, accuracies, losses, consensus_error=0, test_sizes=1, metrics=None):
        """Set testing results uploaded from nodes.

        Args:
            accuracies (dict[float]): Testing accuracies of nodes.
            losses (dict[float]): Testing losses of nodes.
            consensus_error (float): The nodes models consensus error.
            test_sizes (dict[float]): Test dataset sizes of nodes.
            metrics (dict): Node testing metrics.
        """
        self.set_node_uploads(metric.TEST_ACCURACY, accuracies)
        self.set_node_uploads(metric.TEST_LOSS, losses)
        self.set_node_uploads(metric.TEST_DATA_SIZE, test_sizes)
        self.set_node_uploads(metric.TEST_CONSENSUS_ERROR, consensus_error)

    def set_node_avg_model_uploads_test(self, test_avg_model_accuracies, test_avg_model_losses):
        """
         Set testing results on controller avg model uploaded from nodes.

        Args:
            test_avg_model_accuracies (dict[float]): Testing accuracies of nodes on controller average model.
            test_avg_model_losses (dict[float]): Testing losses of nodes on controller average model.
        """
        self.set_node_uploads(metric.TEST_AVG_MODEL_ACCURACY, test_avg_model_accuracies)
        self.set_node_uploads(metric.TEST_AVG_MODEL_LOSS, test_avg_model_losses)

    def set_node_uploads(self, key, value):
        """A general function to set uploaded content from nodes.

        Args:
            key (str): Dictionary key.
            value (*): Uploaded content.
        """
        self._node_uploads[key] = value

    def get_node_uploads(self):
        """Get node uploaded contents.

        Returns:
            dict: A dictionary that contains node uploaded contents.
        """
        return self._node_uploads

    def _do_every(self, every_epoch, every_iteration, current_round, rounds):
        every = every_epoch if self.conf.node.epoch_or_iteration == "epoch" else every_iteration
        return current_round % every == 0 or current_round == rounds or current_round == 0

    def should_print(self):
        if self.conf.node.epoch_or_iteration == "iteration":
            if self._current_round == 0:
                return True
            if self._current_round % self.conf.controller.print_interval != 0:
                return False
        return True

    def print_(self, content):
        """print only the controller should print.

        Args:
            content (str): The content to log.
        """
        if self.should_print():
            logger.info(content)

    # Functions for tracking
    def init_train_test_track(self):
        """Initialize the training performance metric."""
        for node in self._nodes:
            self._train_accuracies[node.cid] = [0]
            self._train_losses[node.cid] = [0]
            self._test_accuracies[node.cid] = []
            self._test_losses[node.cid] = []
            self._test_avg_model_accuracies[node.cid] = []
            self._test_avg_model_losses[node.cid] = []
            if self.conf.node.calculate_static_regret:
                self._train_regrets[node.cid] = [0]
        self._train_accuracies["avg"] = [0]
        self._train_losses["avg"] = [0]
        self._test_accuracies["avg"] = []
        self._test_losses["avg"] = []
        self._test_avg_model_accuracies["avg"] = []
        self._test_avg_model_losses["avg"] = []
        if self.conf.node.calculate_static_regret:
            self._train_regrets["avg"] = [0]

    def save_train_test_data(self):
        record = {'dataset': self.conf.data.dataset,
                  'honest_size': self.graph_.honest_size,
                  'byzantine_size': self.graph_.byzantine_size,
                  'conf': self.conf,
                  metric.TRAIN_LOSS: self._train_losses,
                  metric.TRAIN_ACCURACY: self._train_accuracies,
                  metric.TRAIN_STATIC_REGRET: self._train_regrets,
                  metric.TEST_LOSS: self._test_losses,
                  metric.TEST_ACCURACY: self._test_accuracies,
                  metric.TEST_CONSENSUS_ERROR: self._test_consensus_errors,
                  metric.TEST_AVG_MODEL_ACCURACY: self._test_avg_model_accuracies,
                  metric.TEST_AVG_MODEL_LOSS: self._test_avg_model_losses,
                  metric.CUMULATIVE_TEST_ROUND: self._cumulative_test_round}

        if self.conf.controller.record_in_file:
            title = self.title + ".pkl"
            path_list = [self.conf.controller.record_root, self.conf.data.dataset, self.conf.model,
                         self.conf.controller.aggregation_rule]
            dump_file_in_root(title, record, path_list=path_list, root="")

    def init_title(self):
        """Initialize save title."""
        centralized = "centralized" if self.conf.graph.centralized else "decentralized"
        end_round = "epoch{}".format(self.conf.controller.rounds) if self.conf.node.epoch_or_iteration == "epoch" \
            else "iteration{}".format(self.conf.controller.rounds_iterations)

        self.title = "{}_{}_{}_h{}_b{}_{}_{}_{}_lr{}_{}_mo{}_{}".format(
            self.conf.controller.attack_type[:4],
            self.conf.graph.graph_type[:3], centralized[:3],
            self.graph_.honest_size,
            self.graph_.byzantine_size, end_round,
            self.conf.task_name,
            self.conf.node.lr_controller[:2] + self.conf.node.lr_controller[-4:-2],
            self.conf.lr_controller_param.init_lr,
            self.conf.node.momentum_controller[:2] + self.conf.node.momentum_controller[-4:-2],
            self.conf.lr_controller_param.init_momentum,
            self.conf.data.partition_type
        )

    def track_train_results(self, train_time):
        """
        Record each node train accuracy, loss and regret to the self._.
        """
        train_node_num = 0
        acc = 0
        losses = 0
        regrets = 0
        train_metric = self._node_uploads[metric.NODE_METRICS]
        for node in self._nodes:
            if node in self.selected_nodes and node.cid in self.graph_.honest_nodes:
                train_node_num += 1
                self._train_accuracies[node.cid].append(train_metric[node.cid][metric.TRAIN_ACCURACY])
                self._train_losses[node.cid].append(train_metric[node.cid][metric.TRAIN_LOSS])
                acc += train_metric[node.cid][metric.TRAIN_ACCURACY]
                losses += train_metric[node.cid][metric.TRAIN_LOSS]
                if self.conf.node.calculate_static_regret:
                    self._train_regrets[node.cid].append(train_metric[node.cid][metric.TRAIN_STATIC_REGRET])
                    regrets += train_metric[node.cid][metric.TRAIN_STATIC_REGRET]
            else:
                self._train_accuracies[node.cid].append([0])
                self._train_losses[node.cid].append([0])
                if self.conf.node.calculate_static_regret:
                    self._train_regrets[node.cid].append([0])

        self._train_node_num.append(train_node_num)
        self._train_accuracies["avg"].append(acc / train_node_num)
        self._train_losses["avg"].append(losses / train_node_num)
        if self.conf.node.calculate_static_regret:
            self._train_regrets["avg"].append(regrets)

        if self.conf.node.calculate_static_regret:
            self.print_('Train time {:.2f}s, Train loss: {:.6f}, Train accuracy: {:.2f}%, Train regret: {:6f}'.format(
                train_time, self._train_losses["avg"][-1], self._train_accuracies["avg"][-1] * 100,
                self._train_regrets["avg"][-1]))
        else:
            self.print_('Train time {:.2f}s, Train loss: {:.6f}, Train accuracy: {:.2f}%'.format(
                train_time, self._train_losses["avg"][-1], self._train_accuracies["avg"][-1] * 100))

    def track_test_results(self, test_time):
        """
        Record each node train accuracy, loss and regret to the self._.
        """
        test_node_num = 0
        acc = 0
        losses = 0
        acc_avg_model = 0
        losses_avg_model = 0
        test_metric = self.get_node_uploads()
        self._cumulative_times.append(time.time() - self._start_time)
        test_node = self.get_test_nodes()
        for node in self._nodes:
            if node in self.selected_nodes and node.cid in self.graph_.honest_nodes:
                test_node_num += 1
                self._test_accuracies[node.cid].append(test_metric[metric.TEST_ACCURACY][node.cid])
                self._test_losses[node.cid].append(test_metric[metric.TEST_LOSS][node.cid])
                self._test_avg_model_accuracies[node.cid].append(test_metric[metric.TEST_AVG_MODEL_ACCURACY][node.cid])
                self._test_avg_model_losses[node.cid].append(test_metric[metric.TEST_AVG_MODEL_LOSS][node.cid])
                acc += test_metric[metric.TEST_ACCURACY][node.cid]
                losses += test_metric[metric.TEST_LOSS][node.cid]
                acc_avg_model += test_metric[metric.TEST_AVG_MODEL_ACCURACY][node.cid]
                losses_avg_model += test_metric[metric.TEST_AVG_MODEL_LOSS][node.cid]
            elif node in test_node:
                self._test_accuracies[node.cid].append(test_metric[metric.TEST_ACCURACY][node.cid])
                self._test_losses[node.cid].append(test_metric[metric.TEST_LOSS][node.cid])
                self._test_avg_model_accuracies[node.cid].append(test_metric[metric.TEST_AVG_MODEL_ACCURACY][node.cid])
                self._test_avg_model_losses[node.cid].append(test_metric[metric.TEST_AVG_MODEL_LOSS][node.cid])
            else:
                self._test_accuracies[node.cid].append(0)
                self._test_losses[node.cid].append(0)
                self._test_avg_model_accuracies[node.cid].append(0)
                self._test_avg_model_losses[node.cid].append(0)

        self._test_node_num.append(test_node_num)
        self._test_accuracies["avg"].append(acc / test_node_num)
        self._test_losses["avg"].append(losses / test_node_num)
        self._test_avg_model_accuracies["avg"].append(acc_avg_model / test_node_num)
        self._test_avg_model_losses["avg"].append(losses_avg_model / test_node_num)
        self._test_consensus_errors.append(test_metric[metric.TEST_CONSENSUS_ERROR])
        self._cumulative_test_round.append(self._current_round)

        if self.controller_retain_one_model:
            self.print_('Test time {:.2f}s, Test loss: {:.6f}, Test accuracy: {:.2f}%'.format(
                test_time, self._test_losses["avg"][-1], self._test_accuracies["avg"][-1] * 100))
        else:
            self.print_('Test time {:.2f}s, Test loss: {:.6f}, Test accuracy: {:.2f}%, Consensus error: {:.6f}'.format(
                test_time, self._test_losses["avg"][-1], self._test_accuracies["avg"][-1] * 100,
                self._test_consensus_errors[-1]))
            self.print_('AVG Model Test loss: {:.6f}, Test accuracy: {:.2f}%'.format(
                self._test_avg_model_losses["avg"][-1], self._test_avg_model_accuracies["avg"][-1] * 100))
        self.wandb_log()

    def wandb_log(self):
        """
        If self.conf.wandb_param.use_wandb is True, we use this method to show train process.
        """
        if self.should_print() and self.conf.wandb_param.use_wandb:
            if self._current_round != 0:
                self.wandb_message["Train accuracy"] = self._train_accuracies["avg"][-1] * 100
                self.wandb_message["Train loss"] = self._train_losses["avg"][-1]
            self.wandb_message["Test accuracy"] = self._test_accuracies["avg"][-1] * 100
            self.wandb_message["Test loss"] = self._test_losses["avg"][-1]
            if self.conf.node.calculate_static_regret:
                self.wandb_message["Train regret"] = self._train_regrets["avg"][-1]
            if not self.controller_retain_one_model:
                self.wandb_message["Consensus error"] = self._test_consensus_errors[-1]
                self.wandb_message["AVG Model Test loss"] = self._test_avg_model_losses["avg"][-1]
                self.wandb_message["AVG Model Test accuracy"] = self._test_avg_model_accuracies["avg"][-1] * 100
            wandb.log(self.wandb_message)

