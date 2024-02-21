import torch

__all__ = ['make_dataset', "trans_torch", "make_mnist", "make_cifar10", "make_cifar100", "make_synthetic",
           "FEATURE_TYPE", "TARGET_TYPE", "TARGET_TYPE_SYN_CONS", "TARGET_TYPE_SYN_CLASS",
           "SYN_NUM_CLASSES", "SYN_DATA_DISTRIBUTION_NUM", "TARGET_TYPE_SYN"]

FEATURE_TYPE = torch.float32
TARGET_TYPE = torch.int16
"""When SYN_NUM_CLASSES=1, mean our synthetic data label is successive, we let label class is float"""
TARGET_TYPE_SYN_CONS = torch.float32
"""When SYN_NUM_CLASSES > 1, mean our synthetic data label is classified, we let label class is int"""
TARGET_TYPE_SYN_CLASS = torch.int16
VALUE_TYPE = torch.float32

""" Default 1 mean synthetic data label is successive, 
other positive integers indicate that there are multiple categories"""
SYN_NUM_CLASSES = 1  # 10

"""When SYN_NUM_CLASSES=1, SYN_DATA_DISTRIBUTION_NUM=x mean the 
 continuous synthetic data consists of x distributions, x is positive integer.
 When SYN_NUM_CLASSES>1, SYN_DATA_DISTRIBUTION_NUM=SYN_NUM_CLASSES, means the SYN_NUM_CLASSES classified synthetic data.
 """
SYN_DATA_DISTRIBUTION_NUM = 10

# If synthetic is continuous, then TARGET_TYPE_SYN equal to TARGET_TYPE_SYN_CONS, else TARGET_TYPE_SYN_CLASS
TARGET_TYPE_SYN = TARGET_TYPE_SYN_CONS if SYN_NUM_CLASSES == 1 else TARGET_TYPE_SYN_CLASS
