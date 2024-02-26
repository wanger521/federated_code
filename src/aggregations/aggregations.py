import copy
import itertools
import math

import torch

from src.aggregations.base_aggregation import DistributedAggregation
from src.datas.make_data import FEATURE_TYPE
from scipy import stats
from geom_median.torch import compute_geometric_median
from src.library.logger import create_logger

logger = create_logger()


class Mean(DistributedAggregation):
    """
    Mean aggregation.
    Args:
        graph (graph): the class graph, including  CompleteGraph, ErdosRenyi, TwoCastle, RingCastle, OctopusGraph.
        If graph is centralized, we use CompleteGraph for default.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        self.conf = conf
        super(Mean, self).__init__(name='mean',
                                   graph=graph)

    def run_one_node(self, all_messages, selected_nodes_cid, node, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node, selected_nodes_cid)
        return torch.unsqueeze(torch.mean(neighbor_messages, dim=0), dim=0)


class NoCommunication(DistributedAggregation):
    """The node does not communicate with others, this method is not suite for centralized graph."""

    def __init__(self, graph, conf=None, *args, **kwargs):
        self.conf = conf
        super(NoCommunication, self).__init__(name='no_communication',
                                              graph=graph)

    def run_one_node(self, all_messages, selected_nodes_cid, node, *args, **kwargs):
        return torch.unsqueeze(all_messages[node], dim=0)


class MeanWeightMH(DistributedAggregation):
    """Use the Metropolis-Hastings rule to calculate the access degree of the nodes as weights.
    If graph is centralized, this equal the mean aggregation.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        self.conf = conf
        super(MeanWeightMH, self).__init__(name='meanWeightMH', graph=graph)
        self.selected_nodes_cid = list(range(self.graph.node_size))
        self.W = MeanWeightMH.mh_rule(self.graph, self.selected_nodes_cid)


    def run_one_node(self, all_messages, selected_nodes_cid, node, *args, **kwargs):
        self.W = self.W.to(all_messages)
        agg_messages = torch.tensordot(self.W[node], all_messages, dims=[[0], [0]])
        return torch.unsqueeze(agg_messages, dim=0)

    def run_decentralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        self.W = self.W.to(all_messages)
        return torch.tensordot(self.W, all_messages, dims=[[1], [0]])

    def run_centralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, selected_nodes_cid[0],
                                                              selected_nodes_cid)
        return torch.unsqueeze(torch.mean(neighbor_messages, dim=0), dim=0)

    def update_supporting_information(self, new_graph, node, selected_nodes_cid):
        if new_graph is None:
            if len(selected_nodes_cid) != self.graph.node_size:
                if set(selected_nodes_cid).symmetric_difference(set(self.selected_nodes_cid)):
                    self.selected_nodes_cid = selected_nodes_cid
                    self.W = MeanWeightMH.mh_rule(self.graph, self.selected_nodes_cid)
        else:
            if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
                self.selected_nodes_cid = selected_nodes_cid
                self.reset_graph(new_graph)
                self.W = MeanWeightMH.mh_rule(self.graph, self.selected_nodes_cid)

    @staticmethod
    def mh_rule(graph, selected_nodes_cid):
        # Metropolis-Hastings rule
        node_size = graph.number_of_nodes()
        W = torch.ones((graph.node_size, graph.node_size), dtype=FEATURE_TYPE) / (graph.node_size)
        for i in range(node_size):
            if i not in selected_nodes_cid:
                continue
            i_n = len(list(set(graph.neighbors[i]).intersection(set(selected_nodes_cid)))) + 1
            for j in range(node_size):
                if i == j or not graph.has_edge(j, i) or j not in selected_nodes_cid:
                    continue
                j_n = len(list(set(graph.neighbors[j]).intersection(set(selected_nodes_cid)))) + 1
                W[i][j] = 1 / max(i_n, j_n)
                W[i][i] -= W[i][j]
        return W


class Median(DistributedAggregation):
    """Use median to aggregate."""

    def __init__(self, graph, conf=None, *args, **kwargs):
        self.conf = conf
        super(Median, self).__init__(name='median', graph=graph)

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node, selected_nodes_cid)
        return torch.unsqueeze(Median.median(neighbor_messages), dim=0)

    @staticmethod
    def median(neighbor_messages):
        return neighbor_messages.median(dim=0).values


class GeometricMedian(DistributedAggregation):
    """
    Use geometric median to robust calculate.
    Note:
        If the parameter use_others_geo_med is True, we use the package compute_geometric_median to calculate
        the geometric median, else we use our way to calculate.
    """

    def __init__(self, graph, conf=None, max_iter=80, eps=1e-5, *args, **kwargs):
        super(GeometricMedian, self).__init__(name='geometric_median',
                                              graph=graph)
        self.max_iter = max_iter
        self.eps = eps
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node, selected_nodes_cid)
        return torch.unsqueeze(GeometricMedian.geometric_median(neighbor_messages=neighbor_messages,
                                                                max_iter=self.max_iter, eps=self.eps), dim=0)

    @staticmethod
    def geometric_median(neighbor_messages, max_iter=80, eps=1e-5, use_others_geo_med=False):
        """Choose the method which we use, if the parameter use_others_geo_med is True, we use the
         package compute_geometric_median to calculate the geometric median, else we use our way to calculate."""
        # TODO: package compute_geometric_median runs error: the tensors some in cuda, some in cpu.
        if use_others_geo_med is True:
            return compute_geometric_median(points=neighbor_messages, eps=eps, maxiter=max_iter).median

        guess = torch.mean(neighbor_messages, dim=0)
        for _ in range(max_iter):
            dist_li = torch.norm(neighbor_messages - guess, dim=1)
            # for i in range(len(dist_li)):
            #     if dist_li[i] == 0:
            #         dist_li[i] = 1
            # temp1 = torch.sum(torch.stack(
            #     [w / d for w, d in zip(neighbor_messages, dist_li)]), dim=0)
            # temp2 = torch.sum(1 / dist_li)

            dist_li = torch.clamp(dist_li, min=1e-8)
            temp1 = torch.sum(neighbor_messages / dist_li[:, None], dim=0)
            temp2 = torch.sum(1 / dist_li)
            guess_next = temp1 / temp2
            guess_movement = torch.norm(guess - guess_next)
            guess = guess_next
            if guess_movement <= eps:
                break
        return guess


class Krum(DistributedAggregation):
    """
    Krum aggregation.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(Krum, self).__init__(name='Krum', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node, selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        return torch.unsqueeze(Krum.krum(neighbor_messages=neighbor_messages, byzantine_size=byzantine_size), dim=0)

    @staticmethod
    def krum_index(neighbor_messages, byzantine_size):
        node_size = neighbor_messages.size(0)
        dist = torch.zeros(node_size, node_size, dtype=FEATURE_TYPE)
        for i in range(node_size):
            for j in range(i):
                distance = (neighbor_messages[i].data - neighbor_messages[j].data).norm() ** 2
                # We need minimized distance, so we add a minus sign here
                distance = -distance
                dist[i][j] = distance.data
                dist[j][i] = distance.data
        # The distance from any node to itself must be 0.00, so we add 1 here
        k = node_size - byzantine_size - 2 + 1
        top_v, _ = dist.topk(k=k, dim=1)
        scores = top_v.sum(dim=1)
        return scores.argmax()

    @staticmethod
    def krum(neighbor_messages, byzantine_size):
        index = Krum.krum_index(neighbor_messages, byzantine_size)
        return neighbor_messages[index]


class MKrum(DistributedAggregation):
    """
    Multi Krum aggregation.
    """

    def __init__(self, graph, conf=None, krum_m=2, *args, **kwargs):
        super(MKrum, self).__init__(name='MultiKrum', graph=graph)
        self.krum_m = krum_m
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node, selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        return torch.unsqueeze(MKrum.m_krum(neighbor_messages=neighbor_messages,
                                            byzantine_size=byzantine_size, m=self.krum_m), dim=0)

    @staticmethod
    def m_krum(neighbor_messages, byzantine_size, m=2):
        remain = neighbor_messages
        result = torch.zeros_like(neighbor_messages[0], dtype=FEATURE_TYPE)
        for _ in range(m):
            res_index = Krum.krum_index(remain, byzantine_size)
            result += remain[res_index]
            remain = remain[torch.arange(remain.size(0)) != res_index]
        return result / m


class TrimmedMean(DistributedAggregation):
    """
    Trimmed mean aggregation.
      exact_byz_cnt: True, False
          The parameter of TrimmedMean. True mean use the true byzantine neighbor to trim the messages.
      byz_cnt: -1, 0, 1, 2...
          The parameter of TrimmedMean. When exact_byz_cnt is False, use byz_cnt to control the trim number.
          byz_cnt=-1 means use the Maximum for all Byzantine neighbours. This is for the decentralized graph.
          For normal, byz_cnt is the number of one side to be cut.

    By default, the two double actual number of Byzantines will be trimmed.
    """

    def __init__(self, graph, conf=None, exact_byz_cnt=True, byz_cnt=-1, *args, **kwargs):
        super(TrimmedMean, self).__init__(name='trimmed_mean', graph=graph)
        self.exact_byz_cnt = exact_byz_cnt
        self.byz_cnt = byz_cnt
        self.estimate_byz_cnt_list = [0 for _ in range(self.graph.node_size)]
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.all_neighbor_messages(all_messages, node, selected_nodes_cid)
        byzantine_size = self.estimate_byz_cnt_list[node]
        if 2 * byzantine_size >= len(neighbor_messages):
            byzantine_size = len(neighbor_messages) // 2 - 1
        agg_messages = TrimmedMean.trimmed_mean(neighbor_messages=neighbor_messages,
                                                byzantine_size=byzantine_size)
        agg_messages = agg_messages.to(neighbor_messages)
        trimmed_neighbor_size = len(neighbor_messages) - 2 * byzantine_size
        agg_messages = (agg_messages * trimmed_neighbor_size + all_messages[node]) / (trimmed_neighbor_size + 1)
        return torch.unsqueeze(agg_messages, dim=0)

    def run_centralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, selected_nodes_cid[0],
                                                              selected_nodes_cid)
        byzantine_size = self.estimate_byz_cnt_list[0]
        if 2 * byzantine_size >= len(selected_nodes_cid):
            byzantine_size = len(selected_nodes_cid) // 2 - 1
        return torch.unsqueeze(TrimmedMean.trimmed_mean(neighbor_messages=neighbor_messages,
                                                        byzantine_size=byzantine_size), dim=0)

    def update_supporting_information(self, new_graph, node, selected_nodes_cid):
        if new_graph is not None:
            if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
                self.reset_graph(new_graph)
        if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
            self.estimate_byz_cnt_list = TrimmedMean.get_trimmed_number(self.graph, selected_nodes_cid,
                                                                        self.exact_byz_cnt, self.byz_cnt)

    @staticmethod
    def get_trimmed_number(graph, selected_nodes_cid, exact_byz_cnt=True, byz_cnt=2):
        """Calculate the trimmed rate. exact_byz_cnt is True means trimmed rate is equal to byzantine rate.
                If exact_byz_cnt is False, we use byz_cnt to control the trimmed number,
                it should smaller than the half of neighbors."""
        max_byzantine_size = 0
        estimate_byz_cnt_list = [0 for _ in range(graph.node_size)]
        for node in selected_nodes_cid:
            byzantine_size = len(list(set(graph.byzantine_neighbors_and_itself[node]).
                                      intersection(set(selected_nodes_cid))))
            if exact_byz_cnt:
                estimate_byz_cnt_list[node] = byzantine_size
            else:
                if byz_cnt < 0:
                    max_byzantine_size = max(max_byzantine_size, byzantine_size)
                    estimate_byz_cnt_list[node] = max_byzantine_size
                else:
                    estimate_byz_cnt_list[node] = byz_cnt
            if graph.centralized:
                estimate_byz_cnt_list[0] = estimate_byz_cnt_list[node]
                break
        return estimate_byz_cnt_list

    @staticmethod
    def trimmed_mean(neighbor_messages, byzantine_size):
        node_size = neighbor_messages.size(0)
        proportion_to_cut = byzantine_size / node_size
        tm_np = stats.trim_mean(neighbor_messages.cpu(), proportion_to_cut, axis=0)
        return torch.from_numpy(tm_np).to(neighbor_messages)


class RemoveOutliers(DistributedAggregation):
    """Remove Outliers aggregation."""

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(RemoveOutliers, self).__init__(name='remove_outliers',
                                             graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node,
                                                              selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        return torch.unsqueeze(RemoveOutliers.remove_outliers(neighbor_messages=neighbor_messages,
                                                              byzantine_size=byzantine_size), dim=0)

    @staticmethod
    def remove_outliers(neighbor_messages, byzantine_size):
        mean = torch.mean(neighbor_messages, dim=0)
        # remove the largest 'byzantine_size' model
        distances = torch.tensor([
            -torch.norm(model - mean) for model in neighbor_messages
        ])
        node_size = neighbor_messages.size(0)
        remain_cnt = node_size - byzantine_size
        (_, remove_index) = torch.topk(distances, k=remain_cnt)
        return torch.mean(neighbor_messages[remove_index], dim=0)


class Faba(DistributedAggregation):
    """FABA aggregation."""

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(Faba, self).__init__(name='FABA', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node,
                                                              selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        return torch.unsqueeze(Faba.faba(neighbor_messages=neighbor_messages,
                                         byzantine_size=byzantine_size), dim=0)

    @staticmethod
    def faba(neighbor_messages, byzantine_size):
        remain = neighbor_messages
        for _ in range(byzantine_size):
            mean = torch.mean(remain, dim=0)
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain
            ])
            remove_index = distances.argmax()
            remain = remain[torch.arange(remain.size(0)) != remove_index]
        return torch.mean(remain, dim=0)


class Phocas(DistributedAggregation):
    """Phocas aggregation."""

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(Phocas, self).__init__(name='Phocas', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node,
                                                              selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        if 2 * byzantine_size >= len(selected_nodes_cid):
            byzantine_size = len(selected_nodes_cid) // 2 - 1
        return torch.unsqueeze(Phocas.phocas(neighbor_messages=neighbor_messages,
                                             byzantine_size=byzantine_size), dim=0)

    @staticmethod
    def phocas(neighbor_messages, byzantine_size):
        remain = neighbor_messages
        mean = TrimmedMean.trimmed_mean(remain, byzantine_size)
        mean = mean.to(neighbor_messages)
        # remove the largest 'byzantine_size' model
        distances = torch.tensor([
            torch.norm(model - mean) for model in remain
        ])
        for _ in range(byzantine_size):
            remove_index = distances.argmax()
            remain = remain[torch.arange(remain.size(0)) != remove_index]
            distances = distances[torch.arange(distances.size(0)) != remove_index]
        return torch.mean(remain, dim=0)


class IOS(DistributedAggregation):
    """
    IOS aggregation.
    When graph is centralized, it degenerates to Faba.
    When graph is decentralized, if weight_mh is True, means we use mh Double stochastic matrix to aggregate.
        Else, we use the equal weights (for neighbors) matrix aggregation.
    Notes: We use exact_byz_cnt , byz_cnt to control the trim way, is same as TrimmedMean.
    """

    def __init__(self, graph, conf=None, weight_mh=True, exact_byz_cnt=True, byz_cnt=-1, *args, **kwargs):
        super(IOS, self).__init__(name="IOS", graph=graph)
        self.estimate_byz_cnt_list = None
        node_size = graph.number_of_nodes()
        self.exact_byz_cnt = exact_byz_cnt
        self.byz_cnt = byz_cnt
        self.weight_mh = weight_mh
        self.selected_nodes_cid = list(range(self.graph.node_size))
        self.W = MeanWeightMH.mh_rule(self.graph, self.selected_nodes_cid)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.all_neighbor_messages(all_messages, node, selected_nodes_cid)
        byzantine_size = self.estimate_byz_cnt_list[node]
        return torch.unsqueeze(self.ios(neighbor_messages=neighbor_messages, node=node,
                                        node_message=all_messages[node],
                                        estimate_byz_cnt=byzantine_size), dim=0)

    def run_centralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        node = selected_nodes_cid[0]
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node,
                                                              selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        return torch.unsqueeze(Faba.faba(neighbor_messages=neighbor_messages, byzantine_size=byzantine_size), dim=0)

    def update_supporting_information(self, new_graph, node, selected_nodes_cid):
        if new_graph is None:
            if len(selected_nodes_cid) != self.graph.node_size:
                if set(selected_nodes_cid).symmetric_difference(set(self.selected_nodes_cid)):
                    self.selected_nodes_cid = selected_nodes_cid
                    self.W = IOS.update_weight_metric(self.graph, self.selected_nodes_cid, weight_mh=self.weight_mh)
        else:
            if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
                self.reset_graph(new_graph)
                self.selected_nodes_cid = selected_nodes_cid
                self.W = IOS.update_weight_metric(self.graph, self.selected_nodes_cid, weight_mh=self.weight_mh)

        # If graph is decentralized.
        if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]) and \
                not self.graph.centralized:
            self.estimate_byz_cnt_list = TrimmedMean.get_trimmed_number(self.graph, self.selected_nodes_cid,
                                                                        self.exact_byz_cnt, self.byz_cnt)

    @staticmethod
    def update_weight_metric(graph, selected_nodes_cid, weight_mh=True):
        """If weight_mh is True, means we use mh Double stochastic matrix to aggregation the messages.
        Else, we use mean of equal weights aggregation. """
        if graph.centralized:
            W = torch.ones((graph.node_size, graph.node_size), dtype=FEATURE_TYPE) / (graph.node_size)
        else:
            if weight_mh is True:
                W = MeanWeightMH.mh_rule(graph, selected_nodes_cid)
            else:
                W = torch.ones((graph.node_size, graph.node_size), dtype=FEATURE_TYPE) / (graph.node_size)
                max_degree = -1
                for i in range(graph.node_size):
                    if i not in selected_nodes_cid:
                        continue
                    neighbor_degree = len(list(set(graph.neighbors[i]).intersection(set(selected_nodes_cid))))
                    max_degree = max(neighbor_degree+1, max_degree)
                for i in range(graph.node_size):
                    if i not in selected_nodes_cid:
                        continue
                    for j in range(graph.node_size):
                        if i == j or not graph.has_edge(j, i) or j not in selected_nodes_cid:
                            continue
                        W[i][j] = 1 / max_degree
                        W[i][i] -= W[i][j]
        return W

    def ios(self, neighbor_messages, node, node_message, estimate_byz_cnt):
        """Ios aggregation detail, this is for decentralized graph."""
        remain_messages = neighbor_messages
        neighbors_select_node = self.neighbors_select(node=node, selected_nodes_cid=self.selected_nodes_cid)
        remain_weight = self.W[node][neighbors_select_node]
        remain_weight = remain_weight.to(neighbor_messages)

        for _ in range(estimate_byz_cnt):
            mean = torch.tensordot(remain_weight, remain_messages, dims=1)
            mean += self.W[node][node] * node_message
            mean /= (remain_weight.sum() + self.W[node][node])
            # remove the largest 'byzantine_size' model
            distances = torch.tensor([
                torch.norm(model - mean) for model in remain_messages
            ])
            remove_idx = distances.argmax()
            remain_idx = torch.arange(remain_messages.size(0)) != remove_idx
            remain_messages = remain_messages[remain_idx]
            remain_weight = remain_weight[remain_idx]
        res = torch.tensordot(remain_weight, remain_messages, dims=1)
        res += self.W[node][node] * node_message
        res /= remain_weight.sum() + self.W[node][node]
        return res


class Brute(DistributedAggregation):
    """
    Brute aggregation.
    """

    def __init__(self, graph, conf=None, *args, **kwargs):
        super(Brute, self).__init__(name='Brute', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node,
                                                              selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        return torch.unsqueeze(self.brute(neighbor_messages=neighbor_messages,
                                          byzantine_size=byzantine_size), dim=0)

    def pairwise(self, data):
        """ Simple generator of the pairs (x, y) in a tuple such that index x < index y.
        Args:
          data Indexable (including ability to query length) containing the elements
        Returns:
          Generator over the pairs of the elements of 'data'
        """
        n = len(data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                yield data[i], data[j]

    def brute_selection(self, neighbor_messages, f, **kwargs):
        """ Brute rule.
        brute is also called minimum diameter averaging (MDA)
        The code comes from:
        https://github.com/LPD-EPFL/Garfield/blob/master/pytorch_impl/libs/aggregators/brute.py#L32

        Args:
          neighbor_messages Non-empty list of neighbor_messages to aggregate
          f         Number of Byzantine neighbor_messages to tolerate
          ...       Ignored keyword-arguments
        Returns:
          Selection index set
        """
        n = len(neighbor_messages)
        # Compute all pairwise distances
        distances = [0] * (n * (n - 1) // 2)
        for i, (x, y) in enumerate(self.pairwise(tuple(range(n)))):
            distances[i] = neighbor_messages[x].sub(neighbor_messages[y]).norm().item()
        # Select the set of the smallest diameter
        sel_node_set = None
        sel_diam = None
        for cur_iset in itertools.combinations(range(n), n - f):
            # Compute the current diameter (max of pairwise distances)
            cur_diam = 0.
            for x, y in self.pairwise(cur_iset):
                # Get distance between these two neighbor_messages ("magic" formula valid since x < y)
                cur_dist = distances[(2 * n - x - 3) * x // 2 + y - 1]
                # Check finite distance (non-Byzantine gradient must only contain finite coordinates),
                # drop set if non-finite
                if not math.isfinite(cur_dist):
                    break
                # Check if new maximum
                if cur_dist > cur_diam:
                    cur_diam = cur_dist
            else:
                #  Check if new selected diameter
                if sel_node_set is None or cur_diam < sel_diam:
                    sel_node_set = cur_iset
                    sel_diam = cur_diam
        # Return the selected neighbor_messages
        assert sel_node_set is not None, "Too many non-finite neighbor_messages: a non-Byzantine gradient must only " \
                                         "contain finite coordinates "
        return sel_node_set

    def brute(self, neighbor_messages, byzantine_size, **kwargs):
        """ Brute rule.
        Args:
          neighbor_messages Non-empty list of neighbor_messages to aggregate
          f         Number of Byzantine neighbor_messages to tolerate
          ...       Ignored keyword-arguments
        Returns:
          Aggregated gradient
        """
        sel_node_set = self.brute_selection(neighbor_messages, byzantine_size, **kwargs)
        return sum(neighbor_messages[i] for i in sel_node_set).div_(len(neighbor_messages) - byzantine_size)


class Bulyan(DistributedAggregation):
    def __init__(self, graph, conf=None, *args, **kwargs):
        super(Bulyan, self).__init__(name='Bulyan', graph=graph)
        self.conf = conf

    def run_one_node(self, all_messages, selected_nodes_cid, node, *args, **kwargs):
        neighbor_messages = self.neighbor_messages_and_itself(all_messages, node, selected_nodes_cid)
        byzantine_size = len(self.byzantine_neighbors_and_itself_select(node, selected_nodes_cid))
        return torch.unsqueeze(self.bulyan(neighbor_messages=neighbor_messages,
                                           byzantine_size=byzantine_size), dim=0)

    def bulyan(self, neighbor_messages, byzantine_size):
        remain = neighbor_messages
        selected_ls = []
        node_size = neighbor_messages.size(0)
        selection_size = node_size - 2 * byzantine_size
        # Touting situation in case no optional nodes are available.
        if selection_size <= 0:
            selection_size = 1
        for _ in range(selection_size):
            res_index = Krum.krum_index(remain, byzantine_size)
            selected_ls.append(remain[res_index])
            remain = remain[torch.arange(remain.size(0)) != res_index]
        selection = torch.stack(selected_ls)
        m = Median.median(selection)
        dist = -(selection - m).abs()
        k = max(selection_size - 2 * byzantine_size, 1)  # Touting situation in case no optional nodes are available.
        indices = dist.topk(k=k, dim=0)[1]
        if len(neighbor_messages.size()) == 1:
            result = torch.mean(selection[indices], dim=0)
        else:
            result = torch.stack([
                torch.mean(
                    selection[indices[:, d], d], dim=0) for d in range(neighbor_messages.size(1))])
        return result


class CenteredClipping(DistributedAggregation):
    """
    CenteredClipping aggregation.
    When graph is centralized, we use last step message to calculate which to cc.
    When graph is decentralized, we use the node itself message to calculate which to cc.
        weight_mh: True, False. The parameter of IOS, CenteredClipping. If weight_mh is True, means we use mh double
                    stochastic matrix to aggregation the messages. Else, we use mean of equal weights aggregation.
        threshold_selection: "estimation", "true", "parameter". The parameter of CenteredClipping,
                                the threshold choose way.
        threshold: 10, other real number. The parameter of CenteredClipping.
                    When threshold selection is "parameter", we use threshold to control the threshold.
    """

    def __init__(self, graph, conf=None, weight_mh=True, threshold_selection='estimation',
                 threshold=10, *args, **kwargs):
        super(CenteredClipping, self).__init__(name="centered clipping", graph=graph)
        self.selected_nodes_cid = list(range(graph.node_size))
        self.threshold = threshold
        self.threshold_selection = threshold_selection
        self.weight_mh = weight_mh
        self.W = IOS.update_weight_metric(graph=self.graph, selected_nodes_cid=self.selected_nodes_cid,
                                          weight_mh=self.weight_mh)
        self.neighbors_select_list = [x for x in graph.neighbors]
        self.neighbors_select_honest_list = [x for x in graph.honest_neighbors]
        self.neighbors_select_byzantine_list = [x for x in graph.byzantine_neighbors]
        self.memory = None
        self.conf = conf

    def run(self, all_messages, selected_nodes_cid, node=None, new_graph=None, *args, **kwargs):
        self.update_supporting_information(new_graph=new_graph, node=node, selected_nodes_cid=selected_nodes_cid)
        if self.graph.centralized:
            return self.run_centralized(all_messages, selected_nodes_cid)
        else:
            if node is None:  # return all node aggregation message
                return self.run_decentralized(all_messages, selected_nodes_cid)
            else:  # return one node aggregation message
                return self.run_one_node(all_messages, selected_nodes_cid, node)

    def run_one_node(self, all_messages, selected_nodes_cid, node, *args, **kwargs):
        agg_message_node = self.self_centered_clipping(all_messages=all_messages, node=node)
        return torch.unsqueeze(agg_message_node, dim=0)

    def run_decentralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        agg_messages = copy.deepcopy(all_messages)
        for node in selected_nodes_cid:
            agg_messages[node] = copy.deepcopy(self.self_centered_clipping(all_messages=all_messages, node=node))
        return agg_messages

    def run_centralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        node = self.selected_nodes_cid[0]
        return torch.unsqueeze(self.centered_clipping(all_messages=all_messages, node=node), dim=0)

    def update_supporting_information(self, new_graph, node, selected_nodes_cid):
        if new_graph is None:
            if len(selected_nodes_cid) != self.graph.node_size:
                if set(selected_nodes_cid).symmetric_difference(set(self.selected_nodes_cid)):
                    self.update_neighbors_select(selected_nodes_cid)
                    self.W = IOS.update_weight_metric(graph=self.graph, selected_nodes_cid=self.selected_nodes_cid,
                                                      weight_mh=self.weight_mh)
        else:
            if not (self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
                self.reset_graph(new_graph)
                self.update_neighbors_select(selected_nodes_cid)
                self.W = IOS.update_weight_metric(graph=self.graph, selected_nodes_cid=self.selected_nodes_cid,
                                                  weight_mh=self.weight_mh)

    def update_neighbors_select(self, selected_nodes_cid):
        self.selected_nodes_cid = selected_nodes_cid
        for node in self.selected_nodes_cid:
            self.neighbors_select_list[node] = self.neighbors_select(node, self.selected_nodes_cid)
            self.neighbors_select_honest_list[node] = self.honest_neighbors_select(node, self.selected_nodes_cid)
            self.neighbors_select_byzantine_list[node] = \
                self.byzantine_neighbors_select(node, self.selected_nodes_cid)

    def get_threshold_estimate(self, all_messages, node):
        """If threshold selection is estimation."""
        # TODO: some bad results append, need to fix
        # find the bottom-(honest-size) weights as the estimated threshold
        local_model = all_messages[node]
        node_size = all_messages.size(0)
        norm_list = torch.tensor([
            -(all_messages[n] - local_model).norm()
            if n in self.neighbors_select_list[node] and n != node else 1
            for n in range(node_size)
        ])

        honest_size = len(self.neighbors_select_honest_list[node])
        _, bottom_index = norm_list.topk(k=honest_size)
        top_index = [
            n for n in self.neighbors_select_list[node]
            if n not in bottom_index and n != node
        ]
        weighted_avg_norm = sum([
            self.W[node][n] * norm_list[n] for n in bottom_index
        ])
        cum_weight = sum([
            self.W[node][n] for n in top_index
        ])
        return torch.sqrt(weighted_avg_norm / cum_weight)

    def get_true_threshold(self, all_messages, node):
        """If threshold selection is true."""
        # TODO: some bad results append in centralized graph, need to fix
        # find the bottom-(honest-size) weights as the estimated threshold
        if self.graph.centralized:
            local_model = self.memory

            weighted_avg_norm = sum([
                self.W[node][n] * (all_messages[n] - local_model).norm() ** 2
                for n in self.graph.honest_nodes
            ])
            cum_weight = sum([
                self.W[node][n] for n in self.graph.byzantine_nodes
            ])
            return torch.sqrt(weighted_avg_norm / cum_weight)
        else:
            local_model = all_messages[node]

            weighted_avg_norm = sum([
                self.W[node][n] * (all_messages[n] - local_model).norm() ** 2
                for n in self.neighbors_select_honest_list[node]
            ])
            cum_weight = sum([
                self.W[node][n] for n in self.neighbors_select_byzantine_list[node]
            ])
            return torch.sqrt(weighted_avg_norm / cum_weight)

    def centered_clipping(self, all_messages, node):
        """If graph is centralized."""
        if self.memory is None:
            self.memory = torch.zeros_like(all_messages[node])

        if self.threshold_selection == 'estimation':
            threshold = self.get_threshold_estimate(all_messages, node)
        elif self.threshold_selection == 'true':
            threshold = self.get_true_threshold(all_messages, node)
        elif self.threshold_selection == 'parameter':
            threshold = self.threshold
        else:
            raise ValueError('invalid threshold setting')

        diff = torch.zeros_like(self.memory)
        for n in self.neighbors_select_list[node] + [node]:
            model = all_messages[n]
            norm = (model - self.memory).norm()
            if norm > threshold:
                diff += threshold * (model - self.memory) / norm
            else:
                diff += (model - self.memory)
        diff /= (len(self.neighbors_select_list[node]) + 1)
        self.memory = self.memory + diff

        return self.memory

    def self_centered_clipping(self, all_messages, node):
        """If graph is decentralize."""
        if self.threshold_selection == 'estimation':
            threshold = self.get_threshold_estimate(all_messages, node)
        elif self.threshold_selection == 'true':
            threshold = self.get_true_threshold(all_messages, node)
        elif self.threshold_selection == 'parameter':
            threshold = self.threshold
        else:
            raise ValueError('invalid threshold setting')
        local_model = all_messages[node]
        cum_diff = torch.zeros_like(local_model)

        for n in self.neighbors_select_list[node]:
            model = all_messages[n]
            diff = model - local_model
            norm = diff.norm()
            weight = self.W[node][n]
            if norm > threshold:
                cum_diff += weight * threshold * diff / norm
            else:
                cum_diff += weight * diff
        return local_model + cum_diff
