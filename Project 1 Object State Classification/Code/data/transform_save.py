import torch
from torchvision import transforms
import torchvision.transforms as transforms
import torch.utils.data as data
import os, glob
import pickle, json
import numpy as np
import random, collections
from PIL import Image

image = Image.open("0003.jpg").convert('RGB')

transform_list = {
#transforms.RandomHorizontalFlip(),
#transforms.ColorJitter(hue=.5, saturation=.5),
#transforms.RandomRotation(20),
#transforms.GaussianBlur(17),
#224
#transforms.Resize((256)),
#transforms.CenterCrop(224),
transforms.RandomCrop(size = 224),
transforms.ToTensor()

}

transform = transforms.Compose(transform_list)

image_out = transform(image)

img = transforms.ToPILImage()(image_out)
img.save('rand_crop.jpg')