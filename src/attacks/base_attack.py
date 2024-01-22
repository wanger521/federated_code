import copy


class BaseAttack:
    def __init__(self, name, graph):
        self.graph = graph
        self.name = name
        self.selected_byzantine_nodes_cid = graph.byzantine_neighbors_and_itself
        self.selected_honest_nodes_cid = graph.honest_neighbors_and_itself

    def run(self, all_messages, selected_nodes_cid, node=None, new_graph=None, *args, **kwargs):
        """
        Attack all selected_nodes.
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
            if node is None:  # return all node attack message
                return self.run_decentralized(all_messages, selected_nodes_cid)
            else:  # return one node attack message
                return self.run_one_node(all_messages, selected_nodes_cid, node)

    def run_one_node(self, all_messages, selected_nodes_cid, node, new_graph=None, *args, **kwargs):
        """
        Attack the node in selected nodes cid.
        Args:
            all_messages (torch.Tensor): the stack torch of all node model parameters or gradient.
            selected_nodes_cid (list): the cid number of all selected node.
            node (int): The cid number of which node to aggregate.
            new_graph (graph): If new graph is not None, implies that the connectivity graph of the network
            is time-varying, then the graph for updating the network is new graph.
        Returns: base_messages
        """
        raise NotImplementedError

    def run_decentralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        """
        Warning:
        This approach is not recommended, simply for the sake of uniformity with robust aggregation write-ups,
         in which case it is equivalent to the attacker sending the same attack message to all nodes,
         which does not allow for a decentralized differential attack."""
        for node in selected_nodes_cid:
            return self.run_one_node(all_messages=all_messages,
                                     selected_nodes_cid=selected_nodes_cid, node=node)

    def run_centralized(self, all_messages, selected_nodes_cid, *args, **kwargs):
        """
        For centralized graph, the attack only change the message once.
        """
        node = selected_nodes_cid[0]
        return self.run_one_node(all_messages=all_messages, selected_nodes_cid=selected_nodes_cid, node=node)

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

    def reset_graph(self, new_graph):
        """reset graph for time varying graph."""
        self.graph = new_graph

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
