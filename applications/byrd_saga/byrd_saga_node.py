import copy

from src.train.nodes.base_node import BaseNode
import time

import torch

from src.library import learning_rate_controller
from src.library.cache_io import load_file_in_root, load_model_in_root
from src.optimizations.adam_get_grad import AdamGrad
from src.optimizations.sgd_get_grad import SGDGrad
from src.tracking import metric
from src.library.logger import create_logger

logger = create_logger()


class ByrdSagaNode(BaseNode):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(ByrdSagaNode, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        self.gradient_mean = None
        self.gradient_sum = 0
        self.batch_num = 0
        self.gradient_saga_message = None
        self.round_iteration = conf.node.train_data_size_each[self.cid] // conf.node.batch_size
        self.gradient_record = {}

    def train_iteration(self, conf, device=metric.CPU):
        """
        Execute node training for iteration way. This is for Byrd_Saga.

        Args:
            conf (omegaconf.DictConfig): All configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
        self.train_accuracy = []
        self.static_regret = []
        last_regret = self.current_regret
        self.train_data_size = conf.node.batch_size * conf.node.local_iteration
        # update learning rate and momentum rate
        self.lr = self.lr_controller.get_lr(iteration=conf.node.round_id,
                                            local_iteration=0,
                                            epoch=0, local_epoch=0, extra=0)
        optimizer.param_groups[0]["lr"] = self.lr
        for i in range(conf.node.local_iteration):
            # self.batch_num = (conf.node.round_id * conf.node.local_iteration + i - 1) % self.round_iteration
            correct = 0
            (batched_x, batched_y), beg = next(self.train_loader)
            self.batch_num = beg
            if str(batched_y.dtype) == "torch.int32" or str(batched_y.dtype) == "torch.int16":
                batched_y = batched_y.type(torch.LongTensor)
            x, y = batched_x.to(device), batched_y.to(device)
            optimizer.zero_grad()
            out = self.model(x)
            loss = loss_fn(out, y)
            loss.backward()
            self.current_regret = self.calculate_static_regret(last_regret, loss.item(), x, y, loss_fn)
            # If you want to match the momentum update way "m_t = v_t * m_{t-1} + (1-v_t) * g_t" in algorithm
            self.refine_momentum_process(optimizer, conf, 0, i)
            # for upload gradient and manual update gradient descent
            self.gradient_message = optimizer.get_grad()
            self.gradient_saga_message = self.saga_gradient()
            if self.conf.node.message_type_of_node_sent == "gradient":
                if i != conf.node.local_iteration - 1:
                    self.step_(optimizer, self.gradient_saga_message)
            _, y_pred = torch.max(out, -1)
            correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
            current_correct = correct / len(batched_y)
            self.train_loss.append(float(loss.item()))
            self.train_accuracy.append(float(current_correct))
            self.static_regret.append(float(self.current_regret))
            last_regret = self.current_regret
        logger.debug(
            "Node {}, local iteration: {}, loss: {}".format(self.cid, conf.node.local_iteration, self.train_loss[-1]))
        self.train_time = time.time() - start_time
        logger.debug("Node {}, Train Time: {}".format(self.cid, self.train_time))

    def saga_gradient(self):
        """
        get saga gradient.
        """
        new_message = self.gradient_message
        if self.batch_num not in self.gradient_record:
            self.gradient_record[self.batch_num] = torch.zeros_like(self.gradient_message, requires_grad=False)
        if self.gradient_mean is None:
            self.gradient_mean = torch.zeros_like(self.gradient_message, requires_grad=False)
        new_message.add_(self.gradient_record[self.batch_num], alpha=-1)
        gradient_mean = torch.mul(torch.add(self.gradient_message, -self.gradient_record[self.batch_num]),
                                  1 / self.round_iteration)
        gradient_mean.add_(self.gradient_mean)

        self.gradient_record[self.batch_num] = copy.deepcopy(self.gradient_message)
        new_message.add_(self.gradient_mean)  # * self.gradient_sum/self.round_iteration
        self.gradient_mean = copy.deepcopy(gradient_mean)

        return new_message

    def step_(self, optimizer, new_message):
        # only suit for one parameter group, to update momentum gradient into this optimizer
        # TODO byrd+momentum still is waited for achieve, the way we do will cause bad results.
        for param in optimizer.param_groups:
            cumulated_param = 0
            if self.gradient_message is not None:
                for p in param['params']:
                    param_state = optimizer.state[p]
                    if 'momentum_buffer' not in param_state:
                        pass
                    else:
                        param_size = optimizer.state[p]['momentum_buffer'].nelement()
                        beg, end = cumulated_param, cumulated_param + param_size
                        optimizer.state[p]['momentum_buffer'] = self.gradient_message[beg:end]. \
                            view_as(optimizer.state[p]['momentum_buffer'])
                        cumulated_param = end

        cumulated_param = 0
        for _, param in self.model.named_parameters():
            param_size = param.nelement()
            beg, end = cumulated_param, cumulated_param + param_size
            if self.conf.node.message_type_of_node_sent == "model":
                param.data = new_message[beg:end].view_as(param)
            elif self.conf.node.message_type_of_node_sent == "gradient":
                param.data.add_(new_message[beg:end].view_as(param), alpha=-self.lr)
            else:
                raise ValueError('invalid conf.node.message_type_of_node_sent type.')
            cumulated_param = end

    def construct_upload_request(self):
        """Construct node upload request for training updates and testing results.

        Returns:
            :dict: The upload request.
        """
        uploads = dict()
        try:
            if self._is_train:
                uploads[metric.MODEL] = self.compressed_model
                uploads[metric.GRADIENT_MESSAGE] = self.gradient_saga_message if self.gradient_saga_message is not None else 0
                m = {metric.TRAIN_LOSS: self.train_loss[-1],
                     metric.TRAIN_ACCURACY: self.train_accuracy[-1],
                     metric.TRAIN_STATIC_REGRET: self.static_regret[-1],
                     metric.LEARNING_RATE: self.lr, }
                uploads[metric.METRIC] = m
                uploads[metric.TRAIN_DATA_SIZE] = self.train_data_size
            else:
                uploads[metric.TEST_ACCURACY] = self.test_accuracy
                uploads[metric.TEST_LOSS] = self.test_loss
                uploads[metric.TEST_DATA_SIZE] = self.test_data_size
        except KeyError:
            #  When the datasize cannot be got from dataset, default to use equal aggregate
            logger.error("the datasize cannot be got from dataset.")

        uploads["task_id"] = self.conf.node.task_id
        uploads["round_id"] = self.conf.node.round_id
        uploads["node_id"] = self.cid
        return uploads
