import copy
import os
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.utils import parameters_to_vector

from src.library.cache_io import dump_file_in_cache, dump_file_in_root, file_exist, dump_model_in_root
from src.tracking import metric
# from src.tracking.node import init_tracking
from src.library.float import rounding
# from src.library.distributed import gather_value
from src.library.logger import create_logger
from src.train.controller import BaseController

logger = create_logger()


class RSAController(BaseController):
    def __init__(self, conf, test_data=None, val_data=None):
        super(RSAController, self).__init__(conf, test_data, val_data)
        self.gradient_messages = None
        self.model_paradigm = None
        self.con_lr = conf.lr_controller_param.init_lr
        self.controller_current_model_message = None
        self.controller_last_model_message = None

    def if_controller_retain_one_model(self):
        """The RSA models in  different nodes are different, whether in centralized or decentralized model.

         Return:
             bool: True, False
         """
        controller_retain_one_model = False

        return controller_retain_one_model

    def aggregate_and_attack(self):
        """Byzantine nodes send arbitrary adversary messages,
        Controller aggregates trained models and gradients from nodes via rsa.
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

        # get the model parameter and gradient as two nodes_cnt dimension tensor vectors.
        # The model or gradient for one node is combined as one dimension tensor vector.
        model_messages = []
        gradient_messages = []
        for node_cid, model in enumerate(models):
            model_param = parameters_to_vector([param.data for param in model.parameters()])
            model_messages.append(model_param)
            gradient_param = uploaded_content[metric.GRADIENT_MESSAGE][node_cid]
            gradient_messages.append(gradient_param)
        model_messages = torch.stack(model_messages, dim=0)
        self.gradient_messages = torch.stack(gradient_messages, dim=0)
        self.messages = model_messages
        if self.model_paradigm is None:
            self.model_paradigm = torch.zeros_like(model_messages)

        self.update_con_lr()
        # attack and aggregate
        for node in selected_nodes_cid:
            if not self.conf.controller.byzantine_actually_train and node not in self.graph_.honest_nodes:
                continue
            # attack
            base_messages = self.attack(selected_nodes_cid=selected_nodes_cid, node=node, new_graph=None)
            # rsa aggregate
            gradient_messages[node] = self.aggregate(all_messages=base_messages, selected_nodes_cid=selected_nodes_cid,
                                                     node=node, new_graph=None)

        # Base the aggregation results, update the nodes model
        for node in self.selected_nodes:
            node_cid = node.cid
            self.one_step_gradient_descent_update_or_parameter_update(models[node_cid],
                                                                      gradient_messages[node_cid], self.con_lr)

        self.update_controller_model_message()
        self.set_model(models, load_dict=True)

    def aggregate(self, all_messages, selected_nodes_cid, node, new_graph=None):
        """Aggregate models uploaded from nodes via RSA.

        Args:
            all_messages (tensor): model messages.
            selected_nodes_cid (list[int]): List of select node cid list.
            node (int): the aggregate node cid.
            new_graph (graph): If you want to use the time-varying graph, you can extend with this variable.
        Returns:
            torch.Tensor: rsa gradient message, vector in one dimension.
        """
        lambda_rsa = self.conf.rsa_param.lambda_rsa
        paradigm_num = self.conf.rsa_param.paradigm_num
        if self.controller_last_model_message is None:
            self.controller_last_model_message = copy.deepcopy(all_messages[node])
        self.model_paradigm[node] = self.get_node_model_paradigm(paradigm_num, node, all_messages,
                                                                 self.controller_last_model_message)
        derivative_message = self.get_node_model_derivative(all_messages[node]) if not self.conf.graph.centralized \
            else torch.zeros_like(all_messages[node])
        return self.gradient_messages[node] + lambda_rsa * self.model_paradigm[node] + derivative_message

    def get_node_model_paradigm(self, paradigm_num, node_cid, all_messages, extra_message):
        """
        Get model paradigm_num-paradigm, we only achieve 1-paradigm.
        """
        assert paradigm_num == 1, "We do not achieve other paradigm except 1-paradigm."  # TODO
        results_paradigm = torch.zeros_like(all_messages[0])
        if self.conf.graph.centralized:
            diff = all_messages[node_cid] - extra_message
            results_paradigm = diff.sign()
        else:
            for i in range(len(all_messages)):
                results_paradigm += (all_messages[node_cid] - all_messages[i]).sign()

        return results_paradigm

    def get_node_model_derivative(self, der_message):
        """
        Get the derivative of the square of a two-paradigm number.
        """
        return torch.mul(der_message, 2 * self.conf.rsa_param.weight_decay_rsa)

    def update_controller_model_message(self):
        """
        For centralized RSA, we update x_{0}^{k+1}
        """
        if self.conf.graph.centralized:
            derivative_message = self.get_node_model_derivative(self.controller_last_model_message)
            lambda_rsa = self.conf.rsa_param.lambda_rsa
            controller_paradigm = torch.mul(torch.sum(self.model_paradigm, dim=0), -1)
            update_p = torch.add(derivative_message, torch.mul(controller_paradigm, lambda_rsa))
            self.controller_last_model_message.add_(torch.mul(update_p, -self.con_lr))

    def update_con_lr(self):
        """
        Base on the nodes sent message metric.LEARNING_RATE to update the controller learning rate.
        """
        for key in self.get_node_uploads()[metric.NODE_METRICS]:
            self.con_lr = self.get_node_uploads()[metric.NODE_METRICS][key][metric.LEARNING_RATE]
            break
