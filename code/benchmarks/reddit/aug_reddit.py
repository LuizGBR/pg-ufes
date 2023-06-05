# -*- coding: utf-8 -*-
"""
Autor: André Pacheco
Email: pacheco.comp@gmail.com

Image augmentation classes for PAD-UFES-20 dataset
"""

from imgaug import augmenters as iaa
import numpy as np
import torchvision

class ImgTrainTransform:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.25, iaa.Affine(scale={"x": (1.0, 2.0), "y": (1.0, 2.0)})),
            iaa.Scale(size),
            iaa.Fliplr(0.5),
            iaa.Flipud(0.2),
            iaa.Sometimes(0.25, iaa.Affine(rotate=(-120, 120), mode='symmetric')),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),

            # noise
            iaa.Sometimes(0.1,
                          iaa.OneOf([
                              iaa.Dropout(p=(0, 0.05)),
                              iaa.CoarseDropout(0.02, size_percent=0.25)
                          ])),

            iaa.Sometimes(0.25,
                          iaa.OneOf([
                              iaa.Add((-15, 15), per_channel=0.5), # brightness
                              iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
                          ])),

            # additional augmentations
            iaa.Sometimes(0.5, iaa.CropAndPad(percent=(-0.1, 0.1), pad_mode="constant", pad_cval=(0, 255))),
            iaa.Sometimes(0.5, iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
            iaa.Sometimes(0.5, iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0))),
            iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0.1, 0.5), n_segments=(16, 128))),
            iaa.Sometimes(0.5, iaa.Emboss(alpha=(0.0, 1.0), strength =(0.5, 1.5))),

        ])

    def __call__(self, img):
        img = self.aug.augment_image(np.array(img)).copy()
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)


class ImgEvalTransform:

    def __init__(self, size=(224,224), normalization=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])):

        self.normalization = normalization
        self.size = size

    def __call__(self, img):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.normalization[0], self.normalization[1]),
        ])
        return transforms(img)
