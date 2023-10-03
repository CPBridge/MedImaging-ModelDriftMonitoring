#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import warnings

from pytorch_lightning.utilities.argparse import from_argparse_args
from torchvision import transforms


class Transformer(object):
    pass


class VisionTransformer(Transformer):

    def __init__(self, image_size, normalize="imagenet", channels=3,random_augmentation=False, **kwargs):
        self.image_size = image_size
        self.normalize = normalize
        self.channels = channels
        self.random_augmentation = random_augmentation

    @property
    def normalization(self):
        if self.normalize == "imagenet":
            if self.channels == 3:
                return [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
            else:
                warnings.warn(
                    "ImageNet normalization requires 3 channels, skipping normalization")
        return []

    @property
    def train_transform(self):
        image_transformation = [ 
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ]
        if self.channels == 1:
            image_transformation.append(transforms.Grayscale(num_output_channels=self.channels))
        image_transformation.append(transforms.ToTensor())
        image_transformation += self.normalization
        
        if self.random_augmentation:
            image_transformation += [
                transforms.RandomRotation(degrees=10),  # Random rotation within +/- 10 degrees
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # Random translation up to 5%
                #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Gaussian blurring with sigma between 0.1 to 2.0
                #transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Random lighting and contrast adjustments
            ]
        return transforms.Compose(image_transformation)

    @property
    def val_transform(self):
        image_transformation = [ 
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
        ]
        if self.channels == 1:
            image_transformation.append(transforms.Grayscale(num_output_channels=self.channels))
        image_transformation.append(transforms.ToTensor())
        image_transformation += self.normalization
        
        return transforms.Compose(image_transformation)

    @property
    def infer_transform(self):
        return self.train_transform

    @classmethod
    def add_argparse_args(cls, parser):
        group = parser.add_argument_group("transform")
        group.add_argument("--image_size", type=int, dest="image_size", help="image_size", default=320)
        group.add_argument("--channels", type=int, dest="channels", help="channels", default=3)
        group.add_argument("--normalize", type=str, dest="normalize", help="normalize",
                           default="imagenet", )
        group.add_argument("--random_augmentation", type=bool, dest="random_augmentation", help="Use random augmentation", default=False)

        return parser

    @property
    def dims(self):
        return (self.channels, self.image_size, self.image_size)

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)
