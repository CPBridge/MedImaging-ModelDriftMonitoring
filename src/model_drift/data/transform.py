#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import warnings

from pytorch_lightning.utilities.argparse import from_argparse_args
from torchvision import transforms
from monai.transforms import RandCoarseDropout, RandRotate, RandZoom, RandAffine, RandGaussianSmooth, HistogramNormalize


class Transformer(object):
    pass


class VisionTransformer(Transformer):

    def __init__(self, image_size, normalize="imagenet", channels=3, random_augmentation=False, **kwargs):
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
        image_transformation = []
        if self.channels == 1:
            image_transformation.append(transforms.Grayscale(num_output_channels=self.channels))
        
        image_transformation.extend([ 
            transforms.Resize((320, 320)),
            transforms.CenterCrop((320, 320)),
        ])
        #if self.channels == 1:
        #    image_transformation.append(transforms.Grayscale(num_output_channels=self.channels))
        #image_transformation.append(transforms.ToTensor())
        image_transformation += self.normalization
        image_transformation.append(HistogramNormalize())

        if self.random_augmentation:
            image_transformation += [
                RandRotate(prob=0.5, range_x=0.250),
                RandZoom(prob=0.5, min_zoom=0.9, max_zoom=1.1, padding_mode="constant"),
                RandAffine(prob=0.5, shear_range=(0.2, 0.2), padding_mode="zeros"),
                RandCoarseDropout(prob=0.5, holes=8, spatial_size=16),
                RandGaussianSmooth(prob=0.5, sigma_x=(0.25, 0.75), sigma_y=(0.25, 0.75)),
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
        image_transformation.append(HistogramNormalize())
        return transforms.Compose(image_transformation)

    @property
    def infer_transform(self):
        return self.val_transform

    @classmethod
    def add_argparse_args(cls, parser):
        group = parser.add_argument_group("transform")
        group.add_argument("--image_size", type=int, dest="image_size", help="image_size", default=320)
        group.add_argument("--channels", type=int, dest="channels", help="channels", default=3)
        group.add_argument("--normalize", type=str, dest="normalize", help="normalize",
                           default="imagenet", )
        group.add_argument("--random_augmentation", type=bool, dest="random_augmentation", help="random_augmentation",
                           default=True )

        return parser

    @property
    def dims(self):
        return (self.channels, self.image_size, self.image_size)

    @classmethod
    def from_argparse_args(cls, args, **kwargs):
        return from_argparse_args(cls, args, **kwargs)
