import torch.distributed as dist

import torch


def gather_value(value, world_size, device):
    """Gather the value from devices to a list.

    Args:
        value (float|int): The value to gather.
        world_size (int): The number of processes.
        device (str): The device where the value is on, either cpu or cuda devices.
    Returns:
         list[torch.Tensor]: A list of gathered values.
    """
    v = torch.tensor(value).to(device)
    target = [v.clone() for _ in range(world_size)]
    dist.all_gather(target, v)
    return target
