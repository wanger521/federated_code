import copy
import math
import scipy.stats
import torch
from src.datas.make_data import FEATURE_TYPE

from src.attacks.base_attack import BaseAttack


class NoAttack(BaseAttack):
    """
    No attack.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(NoAttack, self).__init__(name='no_attack', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        return all_messages


class Gaussian(BaseAttack):
    """
    Gaussian attack, change the message by gaussian distribution.
    use_honest_mean: False, True.  The parameter of Gaussian attack, use_honest_mean is True means use the honest
                    neighbors messages means as gaussian distribution mean, and add no other messages.
                    Else, use the parameter mean as gaussian mean. And the byzantine node message.
   mean: 0,1,1.5... The parameter of Gaussian attack, the gaussian distribution mean.
   std: 1,10,50... The parameter of Gaussian attack, the gaussian distribution standard deviation.
    """

    def __init__(self, graph, mean, std, conf=None, use_honest_mean=False, *args, **kwargs):
        super(Gaussian, self).__init__(name='gaussian', graph=graph)
        self.mean = mean
        self.std = std
        self.use_honest_mean = use_honest_mean
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        # base_messages = copy.deepcopy(all_messages)
        base_messages = all_messages
        if self.use_honest_mean:
            self.mean = torch.mean(all_messages[self.selected_honest_nodes_cid[node]], dim=0)
        for by_node in self.selected_byzantine_nodes_cid[node]:
            noise = Gaussian.get_gaussian_noise(base_messages.size(1), self.mean, self.std, base_messages)
            if self.use_honest_mean:
                base_messages[by_node] = noise
            else:
                base_messages[by_node] += noise
        return base_messages

    @staticmethod
    def get_gaussian_noise(tensor_size, mean, std, base_messages):
        noise = torch.randn(tensor_size) * std
        noise = noise.to(base_messages)
        noise += mean
        return noise


class SignFlipping(BaseAttack):
    """
    Sign flipping attack, change the message scale.
    use_honest_mean: False, True.  The parameter of Sign flipping attack, use_honest_mean is True means use the honest
                    neighbors messages means as base sign messages.
                    Else, use the Byzantine message.
    sign_scale: -4, -1, 3... The parameter of sign scale.
    """

    def __init__(self, graph, use_honest_mean, sign_scale=-4, conf=None, *args, **kwargs):
        super(SignFlipping, self).__init__(name='sign_flipping', graph=graph)
        self.use_honest_mean = use_honest_mean
        self.sign_scale = sign_scale
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        mean = None
        # base_messages = copy.deepcopy(all_messages)
        base_messages = all_messages
        if self.use_honest_mean:
            mean = torch.mean(base_messages[self.selected_honest_nodes_cid[node]], dim=0)
        for by_node in self.selected_byzantine_nodes_cid[node]:
            if self.use_honest_mean:
                base_messages[by_node] = self.sign_scale * mean
            else:
                base_messages[by_node] *= self.sign_scale
        return base_messages


class SampleDuplicating(BaseAttack):
    """
    Sample duplicating attack, change the message scale.
    use_honest_mean: False, True.  The parameter of Sample duplicating attack,
                                    use_honest_mean is True means use the honest
                                    neighbors messages means as base duplicating messages.
                                    Else, use the first honest node message.
   sample_scale: 1, 2, 3... The parameter of sample duplicating scale.
    """

    def __init__(self, graph, use_honest_mean, sample_scale=1, conf=None, *args, **kwargs):
        super(SampleDuplicating, self).__init__(name='sample_duplicating', graph=graph)
        self.use_honest_mean = use_honest_mean
        self.sample_scale = sample_scale
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        # base_messages = copy.deepcopy(all_messages)
        base_messages = all_messages
        if self.use_honest_mean:
            attack_message = torch.mean(base_messages[self.selected_honest_nodes_cid[node]], dim=0) * self.sample_scale
        else:
            attack_message = all_messages[self.selected_honest_nodes_cid[node][0]] * self.sample_scale
        for by_node in self.selected_byzantine_nodes_cid[node]:
            base_messages[by_node] = attack_message
        return base_messages


class ZeroValue(BaseAttack):
    """
    Zero value attack, byzantine sent the zero message.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(ZeroValue, self).__init__(name='zero_value', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        # base_messages = copy.deepcopy(all_messages)
        base_messages = all_messages
        attack_message = torch.zeros(base_messages.size(1))
        for by_node in self.selected_byzantine_nodes_cid[node]:
            base_messages[by_node] = attack_message
        return base_messages


class Isolation(BaseAttack):
    """
    Isolation attack, byzantine make the node get zero mean aggregation result..
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(Isolation, self).__init__(name='isolation', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        # base_messages = copy.deepcopy(all_messages)
        base_messages = all_messages
        attack_message = -1 * torch.sum(all_messages[self.selected_honest_nodes_cid[node]], dim=0) / max(
            len(self.selected_byzantine_nodes_cid[node]), 1)
        for by_node in self.selected_byzantine_nodes_cid[node]:
            base_messages[by_node] = attack_message
        return base_messages


class LittleEnough(BaseAttack):
    """
    A little is enough attack, byzantine make the node get zero mean aggregation result.
    little_scale : None, 1, 2,... A little is enough attack scale. None means auto calculate the perfect scale.
                        int means use the little_scale.
    """

    def __init__(self, graph, conf=None, little_scale=None, *args, **kwargs):
        super(LittleEnough, self).__init__(name='little_enough', graph=graph)
        self.little_scale = little_scale
        self.scale_table = [0] * self.graph.node_size
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        # base_messages = copy.deepcopy(all_messages)
        base_messages = all_messages
        mu = torch.mean(all_messages[self.selected_honest_nodes_cid[node]], dim=0)
        std = torch.std(all_messages[self.selected_honest_nodes_cid[node]], dim=0)
        attack_message = mu + self.scale_table[node] * std
        for by_node in self.selected_byzantine_nodes_cid[node]:
            base_messages[by_node] = attack_message
        return base_messages

    def update_supporting_information(self, new_graph, node, selected_nodes_cid):
        """
        For the one interation, only update some information once.
        """
        if new_graph is not None:
            if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
                self.reset_graph(new_graph)

        if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
            for node in selected_nodes_cid:
                self.selected_byzantine_nodes_cid[node] = self.byzantine_neighbors_and_itself_select(node,
                                                                                                     selected_nodes_cid)
                self.selected_honest_nodes_cid[node] = self.honest_neighbors_and_itself_select(node,
                                                                                               selected_nodes_cid)
            if self.little_scale is None:
                self.scale_table = [0] * self.graph.node_size
                for node in selected_nodes_cid:
                    neighbors_size = len(self.neighbors_select(node, selected_nodes_cid))
                    byzantine_size = len(self.byzantine_neighbors_select(node, selected_nodes_cid))
                    s = math.floor((neighbors_size + 1) / 2) - byzantine_size
                    percent_point = (neighbors_size - s) / neighbors_size
                    scale = scipy.stats.norm.ppf(percent_point)
                    self.scale_table[node] = scale
            else:
                self.scale_table = [self.little_scale] * self.graph.node_size


class IPM(BaseAttack):
    """
    IPM attack, same with isolation, but with an epsilon.

    Cong Xie, Oluwasanmi Koyejo, and Indranil Gupta. 2020. Fall of empires:
    Breaking byzantine-tolerant sgd by inner product manipulation. In Uncertainty in
    Artificial Intelligence. PMLR, 261–270.
    """

    def __init__(self, graph, epsilon=0.1, conf=None, *args, **kwargs):
        super(IPM, self).__init__(name='ipm', graph=graph)
        self.conf = conf
        self.epsilon = epsilon

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        # base_messages = copy.deepcopy(all_messages)
        base_messages = all_messages
        attack_message = -self.epsilon * torch.sum(all_messages[self.selected_honest_nodes_cid[node]], dim=0) / max(
            len(self.selected_byzantine_nodes_cid[node]), 1)
        for by_node in self.selected_byzantine_nodes_cid[node]:
            base_messages[by_node] = attack_message
        return base_messages


class AGRFang(BaseAttack):
    """
    Base on unknown aggregation rule to attack.

    Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing. In International Conference
    on Learning Representations.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(AGRFang, self).__init__(name='AGRFang', graph=graph)
        self.conf = conf
        self.agr_scale = self.conf.attacks_param.agr_scale

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        base_messages = all_messages
        honest_neighbors = self.selected_honest_nodes_cid[node]
        mu = torch.mean(all_messages[honest_neighbors], dim=0)
        mu_sign = torch.sign(mu)
        # std = torch.std(local_models[honest_neighbors + [node]], dim=0)
        attack_message = -self.agr_scale * mu_sign + mu
        for by_node in self.selected_byzantine_nodes_cid[node]:
            base_messages[by_node] = attack_message
        return base_messages


class AGRTailored(BaseAttack):
    """
    Base on used aggregation rule to attack.

    Manipulating the byzantine: Optimizing model poisoning attacks and defenses for federated learning.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(AGRTailored, self).__init__(name='AGR-tailored', graph=graph)
        self.conf = conf
        self.agr_scale = self.conf.attacks_param.agr_scale
        self.auto = self.agr_scale = self.conf.attacks_param.auto
        self.aggregation_rule = self.conf.controller.aggregation_rule if self.auto is True \
            else self.conf.attacks_param.against_agg

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        base_messages = all_messages
        if self.aggregation_rule == "TrimmedMean" or self.aggregation_rule == "Median":
            attack_message = self.attack_median_and_trimmed_mean(all_messages, node)
        elif self.aggregation_rule == "SignGuard":
            attack_message = self.attack_sign_guard(all_messages, node)
        else:
            attack_message = self.default_fang(all_messages, node)
        for by_node in self.selected_byzantine_nodes_cid[node]:
            base_messages[by_node] = attack_message
        return base_messages

    def default_fang(self, all_messages, node):
        honest_neighbors = self.selected_honest_nodes_cid[node]
        mu = torch.mean(all_messages[honest_neighbors], dim=0)
        mu_sign = torch.sign(mu)
        attack_message = -self.agr_scale * mu_sign + mu
        return attack_message

    def attack_median_and_trimmed_mean(self, all_messages, node):
        honest_neighbors = self.selected_honest_nodes_cid[node]
        benign_updates = all_messages[honest_neighbors]
        device = benign_updates.device
        mean_grads = benign_updates.mean(dim=0)
        deviation = torch.sign(mean_grads).to(device)
        max_vec, _ = benign_updates.max(dim=0)
        min_vec, _ = benign_updates.min(dim=0)
        b = 2

        neg_pos_mask = torch.logical_and(deviation == -1, max_vec > 0)
        neg_neg_mask = torch.logical_and(deviation == -1, max_vec < 0)
        pos_pos_mask = torch.logical_and(deviation == 1, min_vec > 0)
        pos_neg_mask = torch.logical_and(deviation == 1, min_vec < 0)
        zero_mask = deviation == 0

        # Compute the result for different conditions using tensor operations
        rand_neg_pos = torch.rand(neg_pos_mask.sum(), device=device)
        rand_neg_max = torch.rand(neg_neg_mask.sum(), device=device)
        rand_pos_min = torch.rand(pos_pos_mask.sum(), device=device)
        rand_pos_neg = torch.rand(pos_neg_mask.sum(), device=device)

        neg_pos_max = (
                rand_neg_pos * ((b - 1) * max_vec[neg_pos_mask]) + max_vec[neg_pos_mask]
        )
        neg_neg_max = (
                rand_neg_max * ((1 / b - 1) * max_vec[neg_neg_mask]) + max_vec[neg_neg_mask]
        )
        pos_pos_min = (
                rand_pos_min * ((1 - 1 / b) * min_vec[pos_pos_mask])
                + min_vec[pos_pos_mask] / b
        )
        pos_neg_min = (
                rand_pos_neg * ((1 - b) * min_vec[pos_neg_mask]) + min_vec[pos_neg_mask] * b
        )
        result_zero = mean_grads[zero_mask].repeat(1)

        # Combine the results
        result = torch.zeros_like(mean_grads)
        result[neg_pos_mask] = neg_pos_max
        result[neg_neg_mask] = neg_neg_max
        result[pos_pos_mask] = pos_pos_min
        result[pos_neg_mask] = pos_neg_min
        result[zero_mask] = result_zero

        return result

    def attack_sign_guard(self, all_messages, node):
        honest_neighbors = self.selected_honest_nodes_cid[node]
        benign_updates = all_messages[honest_neighbors]
        device = benign_updates.device
        mean_grads = benign_updates.mean(dim=0)

        # l2norms = [torch.norm(update).item() for update in benign_updates]

        # base_vector = find_orthogonal_unit_vector(mean_grads)
        # return M * base_vector
        num_para = len(mean_grads)
        pos = (mean_grads > 0).sum().item()  # / num_para
        neg = (mean_grads < 0).sum().item()  # / num_para
        zeros = (mean_grads == 0).sum().item()  # / num_para
        noise = torch.hstack(
            [
                torch.rand(pos, device=device),
                -torch.rand(neg, device=device),
                # torch.ones(pos, device=device),
                # -torch.ones(neg, device=device),
                torch.zeros(zeros, device=device),
            ]
        )

        perm = torch.randperm(num_para)

        # Shuffle the vector based on the generated permutation
        update = noise[perm]
        return update
