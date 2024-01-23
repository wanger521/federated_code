from src.datas.make_data import FEATURE_TYPE
from src.datas.make_data.trans_torch import StackedTorchDataPackage
from torchvision import transforms
from torchvision.datasets import CIFAR100


def get_cifar100():
    mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        transforms.RandomCrop(32, padding=4),  # Fill all sides with 0, then crop the image randomly to 32*32
        # 先四周填充0，再把图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
        # transforms.Lambda(lambda x: x / 255)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        transforms.Normalize(mean, std),
    ])

    root = 'data/raw'
    torch_train_dataset = CIFAR100(root=root, train=True,
                                   transform=transform_train, download=True)
    torch_test_dataset = CIFAR100(root=root, train=False,
                                  transform=transform_test, download=True)
    return torch_train_dataset, torch_test_dataset


class MakeCifar100(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('cifar100', get_cifar100)
