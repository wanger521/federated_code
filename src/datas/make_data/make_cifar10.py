from src.datas.make_data import FEATURE_TYPE
from src.datas.make_data.trans_torch import StackedTorchDataPackage
from torchvision import transforms
from torchvision.datasets import CIFAR10


def get_cifar10():
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        transforms.RandomCrop(32, padding=4),  # Fill all sides with 0, then crop the image randomly to 32*32
        # 先四周填充0，再把图像随机裁剪成32*32
        transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=[0.1307],std=[0.3081])
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        # transforms.Normalize(mean=[0.5],std=[0.5])
        # transforms.Lambda(lambda x: x / 255)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=FEATURE_TYPE),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    root = 'data/raw'
    torch_train_dataset = CIFAR10(root=root, train=True,
                                  transform=transform_train, download=True)
    torch_test_dataset = CIFAR10(root=root, train=False,
                                 transform=transform_test, download=True)
    return torch_train_dataset, torch_test_dataset


class MakeCifar10(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('cifar10', get_cifar10)
