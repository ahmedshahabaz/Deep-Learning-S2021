#   1. You have to change the code so that he model is trained on the train set,
#   2. evaluated on the validation set.
#   3. The test set would be reserved for model evaluation by teacher.

from args import get_parser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os, random, math, argparse
import torchvision.transforms as transforms
import models
from dataset import get_loader
from torch.optim.lr_scheduler import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

'''
try:
    from tensorboardX import SummaryWriter
except:
    pass
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = None
train_list = []
val_list = []
# -------------------------------------------

def train(model, epoch, mode, data_loader, loss_func, optimizer, scheduler, train_mean, train_std):
    
    total_loss = 0.0
    total_item = 0

    with tqdm(data_loader, unit="batch") as tepoch:

        for step, (data, label, lenghts, nonZeroRows) in (enumerate(tepoch)):
            
            tepoch.set_description(f"Epoch {epoch},{mode}")
            
            if mode.lower() == "train":
                model.train()
                optimizer.zero_grad()
            
            normalize = transforms.Compose([transforms.Normalize(train_mean, train_std)])
            data = normalize(data.permute(2,0,1)).permute(1,2,0)

            data = data.to(device)
            label = label.to(device)
            
            output = model(data, lenghts)

            loss = loss_func(output[nonZeroRows], label[nonZeroRows])

            if mode.lower() == "train":
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * output[nonZeroRows].shape[0]
            total_item += output[nonZeroRows].shape[0]
            tepoch.set_postfix({"Lss":loss.item()})

        print("Final ", mode , " Loss: ", round(total_loss/total_item, 5))

        return total_loss / total_item

def main(args):
    
    global writer
    global device

    writer = SummaryWriter('./runs/%s'%args.comment)

    full_dataset,train_loader, val_loader, train_mean, train_std = get_loader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, 
        drop_last=False, args=args)

    if torch.cuda.is_available():
        device = torch.device("cuda:"+ str(args.device))

    model = models.myRNN(args)
    #model = model1.myRNN(args)
    params = list(model.parameters())

    #optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, 
    #    lr=args.learning_rate)#, weight_decay = 0.0001)

    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.9,
        weight_decay=0.0001)

    learning_rate_scheduler = None
    #learning_rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=10, 
    #    steps_per_epoch=len(train_loader), epochs=args.epochs)
    #learning_rate_scheduler = MultiStepLR(optimizer, milestones=[2000], 
    #    gamma=0.1, verbose = True)
    #learning_rate_scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor = .2,
     #   verbose=True)
    #learning_rate_scheduler = StepLR(optimizer,:w
     #step_size = 300, gamma = 0.001)
    
    mse_loss = torch.nn.MSELoss()
    model = model.to(device)

    print(model)
    
    writer.add_text('args', " \n".join(['%s %s' % (arg, getattr(args, arg)) for arg in vars(args)]))

    print ("model created & starting training on", device, "...\n\n", )
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params: " ,pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params: ", pytorch_total_params)
    print("---------- --------- ----------")
    print()
    
    min_val_ls = 100000000000000000000

    for epoch in range(args.num_epochs):

        model.train()
        train_loss = train(model = model, epoch = epoch, mode = "Train", data_loader = train_loader,
            loss_func = mse_loss, optimizer = optimizer, scheduler = learning_rate_scheduler,
            train_mean = train_mean, train_std = train_std)

        model.eval()
        with torch.no_grad():
            val_loss = train(model = model, epoch = epoch, mode = "Validation", data_loader = val_loader,
                loss_func = mse_loss, optimizer = optimizer, scheduler = None,
                train_mean = train_mean, train_std = train_std)

        train_list.append(train_loss)
        val_list.append(val_loss)

        if epoch > 1000:

            if val_loss < min_val_ls:

                min_val_ls = val_loss
                directory = "./saved_models/" + args.comment
                if os.path.exists(directory) is False:
                    os.mkdir(directory)
                torch.save(model.state_dict(), directory + "/dl_project_" + str(epoch) + ".pth")

        if args.clip:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

        if learning_rate_scheduler is not None:
            #learning_rate_scheduler.step(train_loss)
            learning_rate_scheduler.step()

        writer.add_scalars(f'',{
            'Ls_trn': train_loss,
            'Ls_val': val_loss,}, epoch)

        print ()

    #np.save('train_loss.npy', np.array([train_list]))
    #np.save('val_loss.npy', np.array([val_list]))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = get_parser(parser)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)