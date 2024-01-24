import copy
import torch
from src.library.logger import create_logger

logger = create_logger()


class DistributedAggregation:
    """
    Father aggregation.
    Args:
        graph (graph): the class graph, including  CompleteGraph, ErdosRenyi, TwoCastle, RingCastle, OctopusGraph.
        If graph is centralized, we use CompleteGraph for default.
        name (str): the name of aggregation.
    """

    def __init__(self, name, graph):
        self.name = name
        self.graph = graph
        # some aggregation may need global state of the system
        # this global state can be store in the global state dictionary
        self.global_state = {}
        self.required_info = set()

    def run(self, all_messages, selected_nodes_cid, node=None, new_graph=None,  *args, **kwargs):
        """
        Aggregate all selected_nodes.
        Args:
            all_messages (torch.Tensor): the stack torch of all node model parameters or gradient.
            selected_nodes_cid (list): the cid number of all selected node.
            new_graph (graph): If new graph is not None, implies that the connectivity graph of the network
            node (int): If node is None, means we aggregate results for all nodes at once. Else,
                    means we aggregate the node neighbor messages for once.
            is time-varying, then the graph for updating the network is new graph.
        """
        self.update_supporting_information(new_graph=new_graph, node=node, selected_nodes_cid=selected_nodes_cid)
        if self.graph.centralized:
            return self.run_centralized(all_messages, selected_nodes_cid)
        else:
            if node is None:  # return all node aggregation message
                return self.run_decentralized(all_messages, selected_nodes_cid)
            else:  # return one node aggregation message
                return self.run_one_node(all_messages, selected_nodes_cid, node)

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        """
        Aggregation the node in selected nodes cid.
        Args:
            all_messages (torch.Tensor): the stack torch of all node model parameters or gradient.
            selected_nodes_cid (list): the cid number of all selected node.
            node (int): The cid number of which node to aggregate.
            new_graph (graph): If new graph is not None, implies that the connectivity graph of the network
            is time-varying, then the graph for updating the network is new graph.
        """
        raise NotImplementedError

    def run_decentralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        """
        For the decentralized graph, all selected nodes need to aggregate once.
        """
        agg_messages = copy.deepcopy(all_messages)
        for node in selected_nodes_cid:
            agg_messages[node] = self.run_one_node(all_messages=all_messages,
                                                   selected_nodes_cid=selected_nodes_cid, node=node)[0]
        return agg_messages

    def run_centralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        """
        For centralized graph, the server only need aggregate once.
        """
        node = selected_nodes_cid[0]
        return self.run_one_node(all_messages=all_messages, selected_nodes_cid=selected_nodes_cid, node=node)

    def update_supporting_information(self, new_graph, node, selected_nodes_cid):
        """
        For the one interation, only update some information once.
        """
        if new_graph is not None:
            if not(self.graph.centralized is False and node is not None and node != selected_nodes_cid[0]):
                self.reset_graph(new_graph)

    def all_neighbor_messages(self, all_messages, node, selected_nodes_cid):
        """The partial messages of the node neighbors."""
        list_neighbor = self.neighbors_select(node=node, selected_nodes_cid=selected_nodes_cid)
        return all_messages[list_neighbor]

    def neighbor_messages_and_itself(self, all_messages, node, selected_nodes_cid):
        """The partial messages of the node neighbors and itself."""
        list_neighbor = self.neighbors_select(node=node, selected_nodes_cid=selected_nodes_cid) + [node]
        return all_messages[list_neighbor]

    def neighbors_select(self, node, selected_nodes_cid):
        """The neighbors of node in selected nodes cid."""
        list_neighbor = list(set(self.graph.neighbors[node]).intersection(set(selected_nodes_cid)))
        list_neighbor.sort()
        return list_neighbor

    def honest_neighbors_select(self, node, selected_nodes_cid):
        """The honest neighbors of node in selected nodes cid."""
        list_neighbor = list(set(self.graph.honest_neighbors[node]).intersection(set(selected_nodes_cid)))
        list_neighbor.sort()
        return list_neighbor

    def byzantine_neighbors_select(self, node, selected_nodes_cid):
        """The byzantine neighbors of node in selected nodes cid."""
        list_neighbor = list(set(self.graph.byzantine_neighbors[node]).intersection(set(selected_nodes_cid)))
        list_neighbor.sort()
        return list_neighbor

    def honest_neighbors_and_itself_select(self, node, selected_nodes_cid):
        """The honest neighbors of node and itself in selected nodes cid."""
        list_neighbor = list(set(self.graph.honest_neighbors_and_itself[node]).intersection(set(selected_nodes_cid)))
        if node not in list_neighbor:
            list_neighbor = list_neighbor + [node]
        list_neighbor.sort()
        return list_neighbor

    def byzantine_neighbors_and_itself_select(self, node, selected_nodes_cid):
        """The byzantine neighbors of node and itself in selected nodes cid."""
        list_neighbor = list(set(self.graph.byzantine_neighbors_and_itself[node]).intersection(set(selected_nodes_cid)))
        list_neighbor.sort()
        return list_neighbor

    def reset_graph(self, new_graph):
        """reset graph for time varying graph."""
        self.graph = new_graph

    def get_subgraph(self, name='', selected_nodes_cid=None):
        """Get subgraph, we do not use this method."""
        subgraph_select = self.graph.subgraph_(name=name, selected_nodes_cid=selected_nodes_cid)
        return subgraph_select



