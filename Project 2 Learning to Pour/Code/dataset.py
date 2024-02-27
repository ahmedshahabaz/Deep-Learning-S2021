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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

class MotionData(data.Dataset):
    """
        Dataloader to generate dataset
    """

    def __init__(self, data_dir="./", npy_file = None ,transforms=None, args=None):
        '''
            function to initialize dataset.
        '''
        self.transform = transformation_functions(args)

        self.data = np.load(data_dir + npy_file)
        self.labels = self.data[:,:,1]
        self.data = np.delete(self.data, 1, axis = 2)


    def __getitem__(self, id):
        """
            Returns a data_item and corresponding label 
        """
        data_item = self.data[id]
        label = self.labels[id]
        
        data_item = self.transform(data_item)
        label = torch.FloatTensor(label)
        data_item = data_item.squeeze()
        nonZeroRows = torch.abs(data_item).sum(dim = 1) > 0
        non_zero_len = len(nonZeroRows[nonZeroRows==True])

        return data_item, label, non_zero_len, nonZeroRows

    def __len__(self):
        return len(self.data)


def get_loader(batch_size, shuffle, num_workers, drop_last=False, args=None):
    '''
        data loader function
    '''
    dataset = MotionData(data_dir=args.data_dir,npy_file = args.npy_file ,args=args)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))


    if args.mode.lower()=='train':# or args.mode=="Train" or args.mode=="TRAIN":

        split = int(np.floor(args.data_split * dataset_size))

        if shuffle:
            np.random.seed(45)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            sampler=train_sampler, num_workers = num_workers)

        validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            sampler=valid_sampler, num_workers = num_workers)
        '''
        Calculating mean and std of train data
        '''
        loader = data.DataLoader(dataset, batch_size= len(train_indices), sampler = train_sampler, 
            num_workers = args.num_workers)

        full_data, _ , _ , nonZeroRows = next(iter(loader))

        full_data = full_data[nonZeroRows]
        mean = full_data.mean(dim = 0)
        std  = full_data.std(dim = 0)

        directory = args.data_dir
        if os.path.exists(directory) is False:
            os.mkdir(directory)
        np.save(directory + "./mean.npy", mean.numpy())
        np.save(directory + "./std.npy", std.numpy())

        print("Train data size: ", len(train_indices))
        print("Validation data size: ", len(val_indices))
        print (" ---------- --------- ---------- \n")

        #full_data = np.load("Robot_Trials_DL.npy")
        #np.save("train.npy", full_data[train_indices])

        return dataset, train_loader, validation_loader, mean, std

    elif args.mode.lower()=='test':

        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
            num_workers = num_workers, shuffle = False)

        mean = np.load("./data/mean.npy")
        std = np.load("./data/std.npy")

        mean = torch.Tensor(mean).to(torch.float64)
        std = torch.Tensor(std).to(torch.float64)

        print("Test data size: ", len(dataset))
        print (" ---------- --------- ---------- \n")
        print()

        return dataset, test_loader, mean, std


"""
   Auxilary data functions
"""

def transformation_functions(args, mode="train"):

    transforms_list = [transforms.ToTensor()]

    return transforms.Compose(transforms_list)
