# importing libraries
import matplotlib.pyplot as plt
import numpy as np
import math

import torch
from torchvision import transforms
import torchvision.transforms as transforms
import torch.utils.data as data
import os, glob
import pickle, json
import numpy as np
import random, collections
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from dataset import *
from args import get_parser
import os, random, math, argparse
import matplotlib.pyplot as plt




def my_plot(axis1, X, Y1, Y2):

	#color = 'tab:red'

	axis1.plot(X, Y1)

	#axis2 = axis1.twinx()

	#color = 'tab:blue'

	#axis2.plot(X, Y2, color = color)


parser = argparse.ArgumentParser()
args = get_parser(parser)

#args.npy_file = 'train.npy'

dataset = MotionData(data_dir=args.data_dir,npy_file = args.npy_file ,args=args)
dataset_size = len(dataset)
indices = list(range(dataset_size))

loader = data.DataLoader(dataset, batch_size= len(indices), 
	num_workers = args.num_workers)

full_data, _ , _ , nonZeroRows = next(iter(loader))

#print(full_data_npy.shape)

mean = np.load("./data/mean.npy")
std = np.load("./data/std.npy")


#full_data = full_data.reshape(585 * 700 , 6)

full_data_npy = np.load(args.data_dir + args.npy_file)
#full_data_npy = full_data_npy.reshape(688 * 700, 7)

print(full_data_npy.shape)

X = np.array(list(range(700)))

#print(X.shape, X)

Y1 = full_data_npy[:,:, 0]
Y2 = full_data_npy[:,:, 1]
#print(X.shape, Y1.shape)
Y3 = full_data_npy[:,:, 2]
#Y4 = np.tanh(X)
  
# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 2)
  
# For Sine Function
my_plot(axis[0,0], Y1[:,1], Y1[:], None)
#my_plot(axis[0,1], X, Y2, None)
#axis[0, 0].set_title("Sine Function")
  
# For Cosine Function
#axis[0, 1].plot(X[nonZeroRows], Y2[nonZeroRows])
#axis[0, 1].set_title("Cosine Function")
  
# For Tangent Function
#axis[1, 0].plot(X[nonZeroRows], Y3[nonZeroRows])
#axis[1, 0].set_title("Tangent Function")
  
# For Tanh Function
#axis[1, 1].plot(X, Y4)
#axis[1, 1].set_title("Tanh Function")
  
# Combine all the operations and display
#plt.imshow(Y1)
plt.show()







