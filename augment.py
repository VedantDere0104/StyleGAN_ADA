from math import degrees
import torch
import torchvision
from torchvision import transforms as T
from torch import nn
from torchvision.transforms.transforms import RandomVerticalFlip
import matplotlib.pyplot as plt
import numpy as np


class CenterCrop(nn.Module):
    def __init__(self , 
                crop_size = 950 , 
                original_image_size = 1024):
        super(CenterCrop , self).__init__()

        self.crop = T.CenterCrop(crop_size)
        self.resize = T.Resize((original_image_size , original_image_size))

    def forward(self , x):
        x = self.crop(x)
        x = self.resize(x)
        return x


class RandomCrop(nn.Module):
    def __init__(self , 
                crop_size = 950 , 
                original_image_size = 1024):
        super(RandomCrop , self).__init__()

        self.crop = T.RandomCrop(crop_size)
        self.resize = T.Resize((original_image_size , original_image_size))

    def forward(self , x):
        x = self.crop(x)
        x = self.resize(x)
        return x

class IdentityTransform(nn.Module):
    def __init__(self):
        super(IdentityTransform , self).__init__()

    def forward(self , x):
        return x

class Augment(nn.Module):
    def __init__(self , 
                vertical_flip = False):
        super(Augment , self).__init__()

        self.transforms = [
            CenterCrop() , 
            T.ColorJitter(brightness=0.5 , hue=0.3) , 
            T.GaussianBlur((5 , 9) , (0.1 , 5)) , 
            T.RandomPerspective(distortion_scale=0.6 , p=1.0) , 
            T.RandomRotation((0 , 180)) , 
            T.Pad(50) , 
            T.RandomAffine((30 , 70) , translate=(0.1 , 0.3) , scale=(0.5 , 0.75)) , 
            RandomCrop() , 
            T.RandomInvert() , 
            T.RandomPosterize(bits=2) , 
            T.RandomSolarize(threshold=192.0) , 
            T.RandomAdjustSharpness(sharpness_factor=2) , 
            T.RandomAutocontrast() , 
            T.RandomEqualize() , 
            T.RandomHorizontalFlip(p=0.9) , 
            T.RandomVerticalFlip(p=0.9) if vertical_flip else IdentityTransform() , 
        ]

    def forward(self , image1 , image2 , p):
        applier = T.RandomApply(self.transforms , p)
        image1 = applier(image1)
        image2 = applier(image2)
        return image1 , image2
        

        