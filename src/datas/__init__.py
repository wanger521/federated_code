import torch

__all__ = ["federated_dataset"]

FEATURE_TYPE = torch.float32
TARGET_TYPE = torch.int16
TARGET_TYPE_SYN = torch.float32  # torch.float64  # torch.int16
VALUE_TYPE = torch.float32
