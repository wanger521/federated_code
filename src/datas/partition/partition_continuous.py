import logging
import random
import numpy as np

from src.datas.make_data.trans_torch import StackedTorchDataPackage, StackedDataSet
import matplotlib.pyplot as plt
from src.datas.partition.partition_unit import Partition
from src.library.logger import create_logger

logger = create_logger()


class HorizontalPartition(Partition):
    def __init__(self, name, partition):
        """
        Parent class according to sample data.
        """
        self.data_distribution = None
        self.partition = partition
        self.data_distribution = self.print_data_distribution()
        super(HorizontalPartition, self).__init__(name, partition)

    def get_subsets(self, dataset):
        return [
            StackedDataSet(features=dataset[p][0], targets=dataset[p][1])
            for i, p in enumerate(self.partition)
        ]

    def print_data_distribution(self):
        """
        Return the distribution of data number and label mean for each node through bar charts.
        """
        data_distribution = {}
        data_y = np.array([label.item() for _, (_, label) in enumerate(self.dataset)])
        for index, data_idx in enumerate(self.partition):
            means, counts = np.mean(data_y[data_idx]), len(data_idx)
            distribution = {"mean": means, "count": counts}
            data_distribution[index] = distribution
        logger.debug("The continuous dataset is divided into distributions at each node as follows: "
                    + str(data_distribution))

        return data_distribution

    def draw_data_distribution(self):
        """
        Draw datas distributions for all node,
        showing the distribution of data number and label mean for each node through cumulative bar charts.
        """

        node_cnt = len(self.partition)
        data_counts = [0 for _ in range(node_cnt)]
        data_means = [0 for _ in range(node_cnt)]
        counts_max, means_max, means_min = 0, 0, 0
        for j in range(node_cnt):
            data_counts[j] = self.data_distribution[j]["count"]
            data_means[j] = self.data_distribution[j]["mean"]
            counts_max = max(counts_max, data_counts[j])
            means_max = max(means_max, data_means[j])
            means_min = min(means_min, data_means[j])

        labels = [i for i in range(node_cnt)]

        plt.rcParams['axes.labelsize'] = 16  # x and y aixs label size
        plt.rcParams['xtick.labelsize'] = 12  # x-axis ticks size
        plt.rcParams['ytick.labelsize'] = 14  # y-axis ticks size
        # plt.rcParams['legend.fontsize'] = 12  # legend size

        # Setting the column spacing
        width = 0.35  # Setting the column width
        x1_list = []
        x2_list = []
        for i in range(len(data_counts)):
            x1_list.append(i)
            x2_list.append(i + width)

        # create figure
        fig, ax1 = plt.subplots()

        # Setting the left Y-axis corresponding to the figure
        ax1.set_ylabel('Data numbers')
        ax1.set_ylim(0, int(counts_max*1.1))
        ax1.bar(x1_list, data_counts, width=width, color='lightseagreen', align='edge')

        # Setting the shared x-axis
        ax1.set_xticks(labels)
        ax1.set_xticklabels(ax1.get_xticklabels())  #
        ax1.set_xlabel('Nodes')
        for a, b, i in zip(x1_list, data_counts, range(len(labels))):
            ax1.text(a+0.2, int(b * 1.03), "%d" % data_counts[i], ha='center')

        # Setting the right Y-axis corresponding to the figure
        ax2 = ax1.twinx()
        ax2.set_ylabel('Data label means')
        ax2.set_ylim(means_min*0.9, means_max*1.1)
        ax2.bar(x2_list, data_means, width=width, color='tab:blue', align='edge', tick_label=labels)

        ax2.set_title(self.name + ' data distribution')

        for a, b, i in zip(x2_list, data_means, range(len(labels))):
            ax2.text(a+0.2, int(b * 1.03), "%.01f" % data_means[i], ha='center')

        plt.tight_layout()
        plt.show()


class EmptyPartition(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw):
        partition = [[] for _ in range(node_cnt)]
        super(EmptyPartition, self).__init__('EmptyPartition', partition)


class SuccessivePartition(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        """
        Successive segmentation divides the dataset to individual nodes.

        This works for datasets with continuous labels as well.

        data separation, with the form of [d(0), d(1), d(2), ..., d(node_cnt)]
        Node i have the dataset indexed by [d(i), d(i+1))

        data partition, with the form of
        [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        Node i have the dataset indexed by [l(n), r(n))

        """
        self.dataset = dataset
        separation = [(i * len(self.dataset)) // node_cnt for i in range(node_cnt + 1)]

        partition = [list(range(separation[i], separation[i + 1]))
                     for i in range(node_cnt)]
        super(SuccessivePartition, self).__init__('SuccessivePartition', partition)


class IIDPartition(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw ) -> None:
        """
        Successive segmentation divides the shuffle dataset to individual nodes

        data separation, with the form of [d(0), d(1), d(2), ..., d(node_cnt)]
        Node i have the dataset indexed by [d(i), d(i+1))

        data partition, with the form of
        [[l(0), r(0)], [l(1), r(1)], ..., [l(n), r(n)]]
        Node i have the dataset indexed by [l(n), r(n))
        """
        self.dataset = dataset
        self.node_cnt = node_cnt
        indexes = list(range(len(dataset)))
        random.shuffle(indexes)
        sep = [(i * len(dataset)) // node_cnt for i in range(node_cnt + 1)]

        partition = [[indexes[i] for i in range(sep[node], sep[node + 1])]
                     for node in range(node_cnt)]
        super(IIDPartition, self).__init__('IIDPartition', partition)


class SharedData(HorizontalPartition):
    def __init__(self, dataset, node_cnt, *args, **kw) -> None:
        self.dataset = dataset
        self.node_cnt = node_cnt
        partition = [list(range(len(dataset)))] * node_cnt
        super(SharedData, self).__init__('SharedData', partition)