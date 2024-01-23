import numpy as np
from src.library.RandomNumberGenerator import RngPackage


class Partition:
    def __init__(self, name, partition):
        self.name = name
        self.partition = partition

    def get_subsets(self, dataset):
        """
        return all subsets of dataset
        ---------------------------------------
        TODO: the partition of data depends on the specific structure
              of dataset.
              In the version, dataset has the structure that all features
              and targets are stacked in tensors. For other datasets with
              different structures, another type of `get_subsets` should
              be implemented.
        """
        raise NotImplementedError

    def __getitem__(self, i):
        return self.partition[i]

    def __len__(self):
        return len(self.partition)


def get_rng_pkg(seed=None):
    """
    get random number generator
    """
    return RngPackage(seed)


def shuffle(data_x, data_y):
    """
    Shuffle data_x and data_y
    """
    num_of_data = len(data_y)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    index = [i for i in range(num_of_data)]
    np.random.shuffle(index)
    data_x = data_x[index]
    data_y = data_y[index]
    return data_x, data_y


def equal_division(num_groups, data_x, data_y=None):
    """Partition data into multiple clients with equal quantity.

    Args:
        num_groups (int): THe number of groups to partition to.
        data_x (list[Object]): A list of elements to be divided.
        data_y (list[Object], optional): A list of data labels to be divided together with the datas.

    Returns:
        list[list]: A list where each element is a list of data of a group/client.
        list[list]: A list where each element is a list of data label of a group/client.

    Example:
        >>> equal_division(3, list[range(9)])
        >>> ([[0,4,2],[3,1,7],[6,5,8]], [])
    """
    if data_y is not None:
        assert (len(data_x) == len(data_y))
        data_x, data_y = shuffle(data_x, data_y)
    else:
        np.random.shuffle(data_x)
    num_of_data = len(data_x)
    assert num_of_data > 0
    data_per_client = num_of_data // num_groups
    large_group_num = num_of_data - num_groups * data_per_client
    small_group_num = num_groups - large_group_num
    split_data_x = []
    split_data_y = []
    for i in range(small_group_num):
        base_index = data_per_client * i
        split_data_x.append(data_x[base_index: base_index + data_per_client])
        if data_y is not None:
            split_data_y.append(data_y[base_index: base_index + data_per_client])
    small_size = data_per_client * small_group_num
    data_per_client += 1
    for i in range(large_group_num):
        base_index = small_size + data_per_client * i
        split_data_x.append(data_x[base_index: base_index + data_per_client])
        if data_y is not None:
            split_data_y.append(data_y[base_index: base_index + data_per_client])

    return split_data_x, split_data_y




