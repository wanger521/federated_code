import os

from torchvision import transforms
from torchvision.datasets import FashionMNIST

from src.datas.make_data import FEATURE_TYPE
from src.datas.make_data.trans_torch import StackedTorchDataPackage


def get_fashion_mnist():
    """
    transforms.Resize((32, 32)), this will make the accuracy decrease,
    but if you want to use model vgg9 or simple_cnn, you need to change the size to 32*32 for suit model.
    """
    transform = transforms.Compose([
        # transforms.Resize((32, 32)),  # same size with cifar10 and cifar100
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        # transforms.Normalize(mean=[0.1307],std=[0.3081])
        transforms.Normalize(mean=[0.5], std=[0.5])
        # transforms.Lambda(lambda x: x / 255)
    ])
    root = 'data/raw'
    torch_train_dataset = FashionMNIST(root=root, train=True,
                                       transform=transform, download=True)
    torch_test_dataset = FashionMNIST(root=root, train=False,
                                      transform=transform, download=True)
    return torch_train_dataset, torch_test_dataset


class MakeFashionMnist(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('fashion_mnist', get_fashion_mnist)
