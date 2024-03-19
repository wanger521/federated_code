from src.datas.make_data import FEATURE_TYPE
from src.datas.make_data.trans_torch import StackedTorchDataPackage
from torchvision import transforms
from torchvision.datasets import ImageNet


def get_image_net():
    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    root = 'data/raw'
    torch_train_dataset = ImageNet(root=root, train=True,
                                   transform=transform_train, download=True)
    torch_test_dataset = ImageNet(root=root, train=False,
                                  transform=transform_test, download=True)
    return torch_train_dataset, torch_test_dataset


class MakeImageNet(StackedTorchDataPackage):
    def __init__(self):
        super().__init__('imageNet', get_image_net)
