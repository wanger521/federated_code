import torch

__all__ = ["federated_dataset"]

FEATURE_TYPE = torch.float64
TARGET_TYPE = torch.int16
TARGET_TYPE_SYN = torch.float64  # torch.float64  # torch.int16
VALUE_TYPE = torch.float64
