import random

import matplotlib.pyplot as plt
import networkx as nx

from src.library.cache_io import get_root_path


def is_valid(nx_graph):
    return nx.connected.is_connected(nx_graph)


class Graph:
    def __init__(self, name, nx_graph, honest_nodes, byzantine_nodes, centralized=True):
        self.name = name
        self.nx_graph = nx_graph
        self.honest_nodes = honest_nodes
        self.byzantine_nodes = byzantine_nodes
        self.centralized = centralized

        # node counting
        self.node_size = nx_graph.number_of_nodes()
        self.honest_size = len(honest_nodes)
        self.byzantine_size = len(byzantine_nodes)
        assert self.honest_size > 0, "Byzantine size should smaller than Node size, node size should bigger than 0."

        # neighbor list
        self.neighbors = [
            list(nx_graph.neighbors(node)) for node in nx_graph.nodes()
        ]
        self.honest_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
             and j in honest_nodes]
            for i in nx_graph.nodes()
        ]
        self.byzantine_neighbors = [
            [j for j in nx_graph.nodes() if nx_graph.has_edge(j, i)
             and j in byzantine_nodes]
            for i in nx_graph.nodes()
        ]
        self.honest_neighbors_and_itself = [
            neighbors + [node] if node in self.honest_nodes else neighbors
            for node, neighbors in enumerate(self.honest_neighbors)
        ]
        self.byzantine_neighbors_and_itself = [
            neighbors + [node] if node in self.byzantine_nodes else neighbors
            for node, neighbors in enumerate(self.byzantine_neighbors)
        ]
        self.neighbors_and_itself = [
            neighbors + [node] for node, neighbors in enumerate(self.neighbors)
        ]
        # neighbor size list
        self.honest_sizes = [
            len(node_list) for node_list in self.honest_neighbors
        ]
        self.byzantine_sizes = [
            len(node_list) for node_list in self.byzantine_neighbors
        ]
        self.neighbor_sizes = [
            self.honest_sizes[node] + self.byzantine_sizes[node]
            for node in nx_graph.nodes()
        ]

        # lost node refers to the node has more than 1/2 byzantine neighbors
        self.lost_nodes = [
            node for node in self.honest_nodes
            if self.honest_sizes[node] <= 2 * self.byzantine_sizes[node]
        ]

    def honest_subgraph(self, name='', relabel=True):
        nx_subgraph = self.subgraph(self.honest_nodes)
        if name == '':
            name = self.name
        if relabel:
            nx_subgraph = nx.convert_node_labels_to_integers(nx_subgraph)
        return Graph(name=name, nx_graph=nx_subgraph,
                     honest_nodes=list(nx_subgraph.nodes()),
                     byzantine_nodes=[], centralized=self.centralized)

    def subgraph_(self, name='', selected_nodes_cid=None):
        if selected_nodes_cid is None:
            selected_nodes_cid = list(range(self.node_size))
        new_honest = list(set(self.honest_nodes).intersection(set(selected_nodes_cid)))
        new_byzantine = list(set(self.byzantine_nodes).intersection(set(selected_nodes_cid)))
        nx_subgraph = self.subgraph(selected_nodes_cid)
        if name == '':
            name = self.name
        nx_subgraph = nx.convert_node_labels_to_integers(nx_subgraph)
        return Graph(name=name, nx_graph=nx_subgraph,
                     honest_nodes=new_honest,
                     byzantine_nodes=new_byzantine)

    def __getattr__(self, attr):
        """
        inherit the properties of 'nx_graph'
        """
        return getattr(self.nx_graph, attr)

    def show(self, show_label=False, show_lost=False, as_subplot=False,
             label_dict=None, node_size=400, font_size=12):
        """
        Show the graph in picture.
        """
        NODE_COLOR_HONEST = '#3399CC'
        NODE_COLOR_BYZANTINE = '#CC3300'
        NODE_COLOR_LOST = '#CCCCCC'
        EDGE_WIDTH = 2

        fig, ax = plt.subplots(1, 1)
        # layout
        pos = nx.kamada_kawai_layout(self.nx_graph)

        # honest nodes
        nx.draw_networkx_nodes(self.nx_graph, pos,
                               node_size=node_size,
                               nodelist=self.honest_nodes,
                               node_color=NODE_COLOR_HONEST,
                               )
        # Byzantine nodes
        nx.draw_networkx_nodes(self.nx_graph, pos,
                               node_size=node_size,
                               nodelist=self.byzantine_nodes,
                               node_color=NODE_COLOR_BYZANTINE,
                               )
        # Lost nodes
        if show_lost:
            nx.draw_networkx_nodes(self.nx_graph, pos,
                                   node_size=node_size,
                                   nodelist=self.lost_nodes,
                                   node_color=NODE_COLOR_LOST,
                                   )

        nx.draw_networkx_edges(self.nx_graph, pos, alpha=0.5, width=EDGE_WIDTH)

        if show_label:
            if label_dict is None:
                label_dict = {
                    i: str(i) for i in range(self.nx_graph.number_of_nodes())
                }
            nx.draw_networkx_labels(self.nx_graph, pos, label_dict,
                                    font_size=font_size)

        ax.axis('off')  # Turn off the border

        if not as_subplot:
            plt.show()

        picture_name = "{}.png".format(self.name)
        path_list = ["record", "report", "picture", "graph"]
        data_root = ""#"../../"
        save_path = get_root_path(picture_name, path_list, data_root, create_if_not_exist=True)
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

    def __getstate__(self):
        state = {
            'name': self.name,
            'nx_graph': self.nx_graph,
            'honest_nodes': self.honest_nodes,
            'byzantine_nodes': self.byzantine_nodes
        }
        return state

    def __setstate__(self, state):
        name = state['name']
        nx_graph = state['nx_graph']
        honest_nodes = state['honest_nodes']
        byzantine_nodes = state['byzantine_nodes']
        self.init(name, nx_graph, honest_nodes, byzantine_nodes)


class CompleteGraph(Graph):
    def __init__(self, node_size, byzantine_size, centralized, conf=None, *args, **kwargs):
        self.conf = conf
        assert node_size > byzantine_size
        graph = nx.complete_graph(node_size)

        honest_nodes = list(range(node_size - byzantine_size))
        byzantine_nodes = list(range(node_size - byzantine_size, node_size))
        name = f'Complete_n={node_size}_b={byzantine_size}'
        super(CompleteGraph, self).__init__(name=name, nx_graph=graph, centralized=centralized,
                                            honest_nodes=honest_nodes,
                                            byzantine_nodes=byzantine_nodes)


class ErdosRenyi(Graph):
    def __init__(self, centralized, node_size, byzantine_size, connected_p=0.7, seed=None, conf=None, *args, **kwargs):
        self.conf = conf
        graph = None
        rng = random if seed is None else random.Random(seed)
        valid = False
        while not valid:
            graph = nx.fast_gnp_random_graph(node_size, connected_p, seed=rng)
            valid = is_valid(graph)

        byzantine_nodes = rng.sample(graph.nodes(), byzantine_size)
        honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        name = f'ER_n={node_size}_b={byzantine_size}_p={connected_p}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super(ErdosRenyi, self).__init__(name=name, nx_graph=graph, centralized=centralized,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes)


class TwoCastle(Graph):
    """
    There are 2k nodes in the network totally
    """

    def __init__(self, centralized, castle_k=5, byzantine_size=1, seed=None, conf=None,*args, **kwargs):
        self.conf = conf
        """castle_k >= 3, byzantine_size <= castle_k-2"""
        assert castle_k >= 3, 'castle_k must be greater than or equal to 3'
        assert byzantine_size <= castle_k - 2, 'byzantine_size must be less than or equal to castle_k - 2'
        node_size = 2 * castle_k
        rng = random if seed is None else random.Random(seed)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_size))
        # inner edges
        for castle in range(2):
            edges_list = [(i, j) for i in range(castle_k * castle, castle_k * castle + castle_k)
                          for j in range(i + 1, castle_k * castle + castle_k)]
            graph.add_edges_from(edges_list)
        # outer edges
        edges_list = [(i, j) for i in range(castle_k)
                      for j in range(castle_k, 2 * castle_k) if i + castle_k != j]
        # edges_list = [(castle_k-1, castle_k)]
        graph.add_edges_from(edges_list)
        byzantine_nodes = rng.sample(graph.nodes(), byzantine_size)
        honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
        name = f'TwoCastle_k={castle_k}_b={byzantine_size}'
        if seed is not None:
            name = name + f'_seed={seed}'
        super(TwoCastle, self).__init__(name=name, nx_graph=graph, centralized=centralized,
                                        honest_nodes=honest_nodes,
                                        byzantine_nodes=byzantine_nodes)


class RingCastle(Graph):
    def __init__(self, castle_cnt, byzantine_size, centralized, seed=None, conf=None, *args, **kwargs):
        self.conf = conf
        name = "RingCastle_castle"
        honest_nodes = []
        byzantine_nodes = []
        node_size = 4 * castle_cnt

        rng = random if seed is None else random.Random(seed)
        graph = nx.Graph()
        graph.add_nodes_from(range(node_size))

        # inner edges
        for castle in range(castle_cnt):
            edges_list = [(i, j) for i in range(4 * castle, 4 * castle + 4)
                          for j in range(i + 1, 4 * castle + 4)]
            graph.add_edges_from(edges_list)
        # outer edges
        for castle in range(castle_cnt):
            next_castle = (castle + 1) % castle_cnt
            graph.add_edges_from([
                (4 * castle + 2, 4 * next_castle + 0),
                (4 * castle + 3, 4 * next_castle + 1),
            ])
            byzantine_nodes = rng.sample(graph.nodes(), byzantine_size)
            honest_nodes = [i for i in graph.nodes() if i not in byzantine_nodes]
            name = f'RingCastle_castle={castle_cnt}_b={byzantine_size}'
            if seed is not None:
                name = name + f'_seed={seed}'
        super(RingCastle, self).__init__(name=name, nx_graph=graph, centralized=centralized,
                                         honest_nodes=honest_nodes,
                                         byzantine_nodes=byzantine_nodes)


class OctopusGraph(Graph):
    def __init__(self, head_cnt, head_byzantine_cnt, hand_byzantine_cnt, centralized, conf=None, *args, **kwargs):
        self.conf = conf
        assert head_cnt > head_byzantine_cnt
        assert head_cnt > hand_byzantine_cnt
        # head
        nx_graph = nx.complete_graph(head_cnt)
        # hands
        nx_graph.add_nodes_from(range(head_cnt, 2 * head_cnt))
        nx_graph.add_edges_from([
            (i, i + head_cnt) for i in range(head_cnt)
        ])

        honest_nodes = list(range(head_byzantine_cnt, head_cnt)) \
                       + list(range(head_cnt + hand_byzantine_cnt, 2 * head_cnt))
        byzantine_nodes = list(range(head_byzantine_cnt)) \
                          + list(range(head_cnt, head_cnt + hand_byzantine_cnt))

        name = f'Octopus_head={head_cnt}_headb={head_byzantine_cnt}_handb={hand_byzantine_cnt}'
        super(OctopusGraph, self).__init__(name=name, nx_graph=nx_graph, centralized=centralized,
                                           honest_nodes=honest_nodes,
                                           byzantine_nodes=byzantine_nodes)
