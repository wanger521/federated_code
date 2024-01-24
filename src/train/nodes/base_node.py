import copy
import time

import torch

from src.library import learning_rate_controller
from src.library.cache_io import load_file_in_root, load_model_in_root
from src.optimizations.adam_get_grad import AdamGrad
from src.optimizations.sgd_get_grad import SGDGrad
from src.tracking import metric
from src.library.logger import create_logger

logger = create_logger()


class BaseNode(object):
    """Default implementation of federated learning node.

    Args:
        cid (str): Node id.
        conf (omegaconf.DictConfig): All configurations.
        train_data (:obj:`FederatedDataset`): Training dataset.
        test_data (:obj:`FederatedDataset`): Test dataset.
        device (str): Hardware device for training, cpu or cuda devices.
        sleep_time (float): Duration of on hold after training to simulate stragglers.


    Override the class and functions to implement customized node.

    Example:
        >>> from src.train.nodes.base_node import BaseNode
        >>> class CustomizedNode(BaseNode):
        >>>     def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        >>>         super(CustomizedNode, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        >>>         pass  # more initialization of attributes.
        >>>
        >>>     def train(self, conf, device=metric.CPU):
        >>>         # Implement customized node training method, which overwrites the default training method.
        >>>         pass
    """

    def __init__(self,
                 cid,
                 conf,
                 train_data,
                 test_data,
                 device,
                 sleep_time=0):
        self.cid = cid
        self.conf = conf
        self.train_data = train_data
        self.train_loader = None
        self.test_data = test_data
        self.test_loader = None
        self.device = device

        self.round_time = 0
        self.train_time = 0
        self.test_time = 0

        self.train_accuracy = []
        self.train_loss = []
        self.train_best_model_loss = []
        self.static_regret = []
        self.test_accuracy = 0
        self.test_loss = 0
        self.current_regret = 0
        self.test_data_size = 1
        self.train_data_size = 1

        self.profiled = False
        self._sleep_time = sleep_time

        self.compressed_model = None
        self.model = None
        self.best_model = None
        self.optimizer = None
        self.lr = None
        self.gradient_message = None
        self.lr_controller = None
        self.momentum_controller = None
        self.init_lr_controller()
        self.init_momentum_controller()

        self._controller_stub = None
        # self._tracker = None
        self._is_train = True

        # if conf.node.track:
        #     self._tracker = init_tracking(init_store=False)

    def run_train(self, model, conf):
        """Conduct training on nodes.

        Args:
            model (nn.Module): Model to train.
            conf (omegaconf.DictConfig): All configurations.
        Returns:
            upload (dict): Training contents.
        """
        self.conf = conf
        self.init_best_model(model)

        self._is_train = True

        self.download(model)

        self.decompression()

        self.pre_train()
        self.train(conf, self.device)
        self.post_train()

        if conf.node.local_test:
            self.test_local()

        self.compression()

        return self.upload()

    def run_test(self, model, conf):
        """Conduct testing on nodes.

        Args:
            model (nn.Module): Model to test.
            conf (omegaconf.DictConfig): All configurations.
        Returns:
            upload (dict): Testing contents.
        """
        self.conf = conf

        self._is_train = False

        self.download(model)

        self.decompression()

        self.pre_test()
        self.test(conf, self.device)
        self.post_test()

        return self.upload()

    def run_avg_model_test(self, model, conf):
        """Conduct testing on nodes.

        Args:
            model (nn.Module): AVG Model to test.
            conf (omegaconf.DictConfig): All configurations.
        Returns:
            upload (dict): Testing contents.
        """
        self._is_train = False

        self.pre_test()
        self.test(conf, self.device, another_model=model)
        self.post_test()

        return self.upload()

    def init_lr_controller(self):
        """Get learning rate controller."""
        lr_controller_class = getattr(learning_rate_controller, self.conf.node.lr_controller)
        self.lr_controller = lr_controller_class(init_lr=self.conf.lr_controller_param.init_lr,
                                                 epoch_or_iteration=self.conf.node.epoch_or_iteration,
                                                 total_epoch=self.conf.controller.rounds,
                                                 total_iteration=self.conf.controller.rounds_iterations,
                                                 decreasing_iter_ls=self.conf.lr_controller_param.decreasing_iter_ls,
                                                 proportion_ls=self.conf.lr_controller_param.proportion_ls,
                                                 final_proportion=self.conf.lr_controller_param.final_proportion,
                                                 a=self.conf.lr_controller_param.a,
                                                 b=self.conf.lr_controller_param.b,
                                                 boundary_iteration=self.conf.lr_controller_param.boundary_iteration,
                                                 boundary_epoch=self.conf.lr_controller_param.boundary_epoch,
                                                 ratio=self.conf.lr_controller_param.ratio,
                                                 multiple_ratio=self.conf.lr_controller_param.multiple_ratio,
                                                 step_interval_interation=self.conf.lr_controller_param.step_interval_interation,
                                                 step_interval_epoch=self.conf.lr_controller_param.step_interval_epoch,
                                                 step_ratio=self.conf.lr_controller_param.step_ratio,
                                                 nodes_cnt=self.conf.graph.nodes_cnt)

    def init_momentum_controller(self):
        """If optimizer is SGD, we use this to init momentum step controller."""
        momentum_controller_class = getattr(learning_rate_controller, self.conf.node.momentum_controller)
        self.momentum_controller = momentum_controller_class(
            init_lr=self.conf.lr_controller_param.init_momentum,
            epoch_or_iteration=self.conf.node.epoch_or_iteration,
            total_epoch=self.conf.controller.rounds,
            total_iteration=self.conf.controller.rounds_iterations,
            decreasing_iter_ls=self.conf.lr_controller_param.decreasing_iter_ls,
            proportion_ls=self.conf.lr_controller_param.proportion_ls,
            final_proportion=self.conf.lr_controller_param.final_proportion,
            a=self.conf.lr_controller_param.a,
            b=self.conf.lr_controller_param.b,
            boundary_iteration=self.conf.lr_controller_param.boundary_iteration,
            boundary_epoch=self.conf.lr_controller_param.boundary_epoch,
            ratio=self.conf.lr_controller_param.ratio,
            multiple_ratio=self.conf.lr_controller_param.multiple_ratio,
            step_interval_interation=self.conf.lr_controller_param.step_interval_interation,
            step_interval_epoch=self.conf.lr_controller_param.step_interval_epoch,
            step_ratio=self.conf.lr_controller_param.step_ratio)

    def init_best_model(self, model):
        """For calculate static regret, we load the online best model."""
        if self.best_model is None:
            title = metric.ONLINE_BEST_MODEL + ".pth"
            path_list = [metric.SAVED_MODELS, self.conf.data.dataset, self.conf.model,
                         metric.MEAN]
            if self.conf.node.calculate_static_regret is True and self.conf.task_name != metric.ONLINE_BEST_MODEL:
                self.best_model = copy.deepcopy(model)
                self.best_model.load_state_dict(load_model_in_root(title, path_list))
                self.best_model.to(self.device)

    def download(self, model):
        """Download model from the controller.

        Args:
            model (nn.Module): Global model distributed from the controller.
        """
        if self.compressed_model:
            self.compressed_model.load_state_dict(model.state_dict())
        else:
            self.compressed_model = copy.deepcopy(model)
        # self.compressed_model = model

    def decompression(self):
        """Decompressed model. It can be further implemented when the model is compressed in the controller."""
        self.model = self.compressed_model

    def pre_train(self):
        """Preprocessing before training."""
        pass

    def train(self, conf, device=metric.CPU):
        """Execute node training.

        Args:
            conf (omegaconf.DictConfig): All configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        if conf.node.epoch_or_iteration == "iteration":
            self.train_iteration(conf, device=device)
        else:
            self.train_epoch(conf, device=device)

    def train_epoch(self, conf, device=metric.CPU):
        """Execute node training for epoch way.

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
        round_iteration = conf.node.train_data_size_each[self.cid] // conf.node.batch_size
        self.train_data_size = round_iteration * conf.node.batch_size * conf.node.local_epoch
        lr = self.lr_controller.get_lr(iteration=0, local_iteration=0,
                                       epoch=conf.node.round_id,
                                       local_epoch=0, extra=0)
        for i in range(conf.node.local_epoch):
            batch_loss = []
            batch_regret = []
            correct = 0
            for j in range(round_iteration):
                batched_x, batched_y = next(self.train_loader)
                if str(batched_y.dtype) == "torch.int32" or str(batched_y.dtype) == "torch.int16":
                    batched_y = batched_y.type(torch.LongTensor)
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                regret = self.calculate_static_regret(last_regret, loss.item(), x, y, loss_fn)
                # update learning rate and momentum rate
                optimizer.param_groups[0]["lr"] = lr
                # If you want to match the momentum update way "m_t = v_t * m_{t-1} + (1-v_t) * g_t" in algorithm
                self.refine_momentum_process(optimizer, conf, i, 0)
                # for upload gradient and manual update gradient descent
                if i == conf.node.local_epoch - 1 and j == round_iteration - 1 \
                        and self.conf.node.message_type_of_node_sent == "gradient":
                    self.gradient_message = optimizer.get_grad()
                    self.lr = optimizer.param_groups[0]["lr"]
                else:
                    optimizer.step()
                # save train loss, accuracy, static regret.
                batch_loss.append(loss.item())
                batch_regret.append(regret)
                last_regret = regret
                # calculate accuracy
                _, y_pred = torch.max(out, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.current_regret = batch_regret[-1]
            current_correct = correct / (round_iteration * conf.node.batch_size)
            self.train_loss.append(float(current_epoch_loss))
            self.train_accuracy.append(float(current_correct))
            self.static_regret.append(float(self.current_regret))
            logger.debug("Node {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
            logger.debug("Node {}, local epoch: {}, accuracy: {}".format(self.cid, i, current_correct))
        self.train_time = time.time() - start_time
        logger.debug("Node {}, Train Time: {}".format(self.cid, self.train_time))

    def train_iteration(self, conf, device=metric.CPU):
        """Execute node training for iteration way.

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
        lr = self.lr_controller.get_lr(iteration=conf.node.round_id,
                                       local_iteration=0,
                                       epoch=0, local_epoch=0, extra=0)
        # update learning rate and momentum rate
        optimizer.param_groups[0]["lr"] = lr
        self.lr = optimizer.param_groups[0]["lr"]
        for i in range(conf.node.local_iteration):
            correct = 0
            batched_x, batched_y = next(self.train_loader)
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
            if i == conf.node.local_iteration - 1 and self.conf.node.message_type_of_node_sent == "gradient":
                self.gradient_message = optimizer.get_grad()
            else:
                optimizer.step()
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

    def post_train(self):
        """Postprocessing after training."""
        pass

    @torch.no_grad()
    def calculate_static_regret(self, last_regret, current_loss, features, targets, loss_fn):
        """
        Calculate the current static regret
        """
        if self.best_model is None:
            return current_loss
        current_regret = last_regret
        with torch.no_grad():
            self.best_model.eval()
            best_loss = loss_fn(self.best_model(features), targets).item()
            regret = current_loss - best_loss
            current_regret += regret
        return current_regret

    def refine_momentum_process(self, optimizer, conf, local_epoch, local_iteration):
        """ If you want to match the momentum update way "m_t = v_t * m_{t-1} + (1-v_t) * g_t" in algorithm ."""
        if self.conf.node.optimizer.type == "SGD" and self.conf.node.optimizer.use_another_momentum and \
                self.conf.node.optimizer.use_momentum:
            optimizer.param_groups[0]["momentum"] = 1 - self.momentum_controller.get_lr(
                iteration=conf.node.round_id,
                local_iteration=local_iteration,
                epoch=conf.node.round_id, local_epoch=local_epoch,
                extra=optimizer.param_groups[0]["lr"])
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.data = (1 - optimizer.param_groups[0]["momentum"]) * p.grad.data

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.model.train()
        self.model.to(device)
        loss_fn = self.load_loss_fn(conf)
        optimizer = self.load_optimizer(conf)
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)
        return loss_fn, optimizer

    def load_loss_fn(self, conf):
        if conf.model == "linear_regression":
            return torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        else:
            return torch.nn.CrossEntropyLoss()

    def load_optimizer(self, conf):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.node.optimizer.type == "Adam":
            if self.optimizer is None:
                optimizer = AdamGrad(self.model.parameters(), lr=conf.lr_controller_param.init_lr)
            else:
                optimizer = self.optimizer
        elif conf.node.optimizer.type == "SGD":
            if self.optimizer is None:
                if self.conf.node.optimizer.use_momentum:
                    optimizer = SGDGrad(self.model.parameters(),
                                        lr=conf.lr_controller_param.init_lr,
                                        momentum=1 - conf.lr_controller_param.init_momentum,
                                        weight_decay=conf.node.optimizer.weight_decay)
                else:
                    optimizer = SGDGrad(self.model.parameters(),
                                        lr=conf.lr_controller_param.init_lr,
                                        weight_decay=conf.node.optimizer.weight_decay)
                self.optimizer = optimizer
            else:
                optimizer = self.optimizer
                # only suit for one parameter group, to update momentum gradient into this optimizer
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
        else:
            raise ValueError('invalid optimizer type, we only achieve SGD and Adam.')
        return optimizer

    def load_loader(self, conf):
        """Load the training data loader.

        Args:
            conf (omegaconf.DictConfig): All configurations.
        Returns:
            partial: Data partial loader.
        """
        return self.train_data

    def test_local(self):
        """Test node local model after training."""
        pass

    def pre_test(self):
        """Preprocessing before testing."""
        pass

    def test(self, conf, device=metric.CPU, another_model=None):
        """Execute node testing.

        Args:
            conf (omegaconf.DictConfig): All configurations.
            device (str): Hardware device for training, cpu or cuda devices.
            another_model: if another_model is None, use self.model.
        """
        begin_test_time = time.time()
        if another_model is None:
            model = self.model
        else:
            model = another_model
        model.eval()
        model.to(device)
        loss_fn = self.load_loss_fn(conf)
        if self.test_loader is None:
            self.test_loader = self.test_data
        # TODO: make evaluation metrics a separate package and apply it here.
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            batched_x, batched_y = next(self.test_loader)
            if str(batched_y.dtype) == "torch.int32" or str(batched_y.dtype) == "torch.int16":
                batched_y = batched_y.type(torch.LongTensor)
            x, y = batched_x.to(device), batched_y.to(device)
            log_probs = model(x)

            loss = loss_fn(log_probs, y)
            _, y_pred = torch.max(log_probs, -1)
            correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
            self.test_loss = loss.item()

            self.test_data_size = len(batched_y)
            self.test_accuracy = float(correct) / self.test_data_size

        logger.debug('Node {}, testing -- Average loss: {:.8f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.cid, self.test_loss, correct, self.test_data_size, self.test_accuracy * 100))

        self.test_time = time.time() - begin_test_time
        model.cpu()

    def post_test(self):
        """Postprocessing after testing."""
        pass

    def compression(self):
        """Compress the node local model after training and before uploading to the controller."""
        self.compressed_model = self.model

    def upload(self):
        """Upload the messages from node to the controller.

        Returns:
            request (dict): up;oad content.
        """
        request = self.construct_upload_request()
        self.post_upload()
        return request

    def post_upload(self):
        """Postprocessing after uploading training/testing results."""
        pass

    def construct_upload_request(self):
        """Construct node upload request for training updates and testing results.

        Returns:
            :dict: The upload request.
        """
        uploads = dict()
        try:
            if self._is_train:
                uploads[metric.MODEL] = self.compressed_model
                uploads[metric.GRADIENT_MESSAGE] = self.gradient_message if self.gradient_message is not None else 0
                m = {metric.TRAIN_LOSS: self.train_loss[-1],
                     metric.TRAIN_ACCURACY: self.train_accuracy[-1],
                     metric.TRAIN_STATIC_REGRET: self.static_regret[-1],
                     metric.LEARNING_RATE:  self.lr}
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

    def simulate_straggler(self):
        """Simulate straggler effect of system heterogeneity."""
        if self._sleep_time > 0:
            time.sleep(self._sleep_time)
