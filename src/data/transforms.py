"""Preprocessing transforms.

Uses ImageNet mean/std because the classifier backbone (DenseNet121) is
ImageNet-pretrained (src/models/classifier.py) — the input distribution needs
to match what the pretrained weights expect.

No horizontal flip augmentation, deliberately: chest X-ray findings are not
left-right symmetric in general (e.g. cardiac silhouette position, situs), so
flipping risks teaching the model an anatomically wrong signal. Only mild,
orientation-preserving augmentation (small rotation, brightness/contrast
jitter) is used for training.
"""
from torchvision import transforms

from src.data.constants import IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int = 224, train: bool = False) -> transforms.Compose:
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return transforms.Compose(ops)
