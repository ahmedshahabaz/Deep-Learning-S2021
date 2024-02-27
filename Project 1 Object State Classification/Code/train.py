#   1. You have to change the code so that he model is trained on the train set,
#   2. evaluated on the validation set.
#   3. The test set would be reserved for model evaluation by teacher.

from args import get_parser
import torch
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import numpy as np
import os, random, math

import model1
import models

from dataset import get_loader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

'''
try:
    from tensorboardX import SummaryWriter
except:
    pass
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------

def train(model, epoch, mode, data_loader, device, loss_func, optimizer, scheduler):
    
    total_correct_preds = 0.0
    total_elements = 1e-10
    loss = 0.0
    #outer = tqdm(total=len(data_loader), desc='Batch', position=0)
    with tqdm(data_loader, unit="batch") as tepoch:

        for step, (image_input, class_idxs, _) in (enumerate(tepoch)):
            
            tepoch.set_description(f"Epoch {epoch},{mode}")

            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)
            
            if (mode == "Train" or mode == "train"):
                model.train()
                optimizer.zero_grad()
            
            output = (model(image_input))#, dim = 1)
            total_elements += output.size(0)
            state_loss = loss_func(output, class_idxs) # --> 32 * 1
            # aggregate loss for logging
            loss += torch.sum(state_loss).item()
        
            # back-propagate the loss in the model & optimize
            if mode == "train" or mode == "Train":
                torch.mean(state_loss).backward()
                optimizer.step()

            # accuracy computation
            #_, pred_idx = torch.max(output, dim=1)
            _, pred_idx = torch.max(F.softmax(output, dim=1), dim=1)
            correct_preds_batch = torch.sum(pred_idx==class_idxs).item()
            total_correct_preds += correct_preds_batch
            #total += output.size(0)
            tepoch.set_postfix({"Acc":round(total_correct_preds/total_elements,2), 
                "Lss":round(loss/total_elements,2)})

    #print('\rEpoch: {}, {} accuracy: {}, loss: {}'.format(epoch,mode, accuracy, loss)) 

    return loss/total_elements, total_correct_preds/total_elements

def main(args):
    
    writer = SummaryWriter('./runs/%s'%args.comment)

    data_loader, dataset = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=True, 
                                           num_workers=args.num_workers, drop_last=False, args=args)

    temp = args.mode
    args.mode = "validation"

    test_loader, test_dataset = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=True, 
                                           num_workers=args.num_workers, drop_last=False, args=args)
    args.mode = temp

    data_size = test_dataset.get_data_size()
    num_classes = test_dataset.get_num_classes()
    instance_size = test_dataset.get_instance_size()
    device = torch.device("cuda:"+ str(args.device))

    model = model1.my_ResNet(pretrained = False, num_classes = num_classes)
    #model = models.my_ResNet(pretrained = False, num_classes = num_classes)
    print(model)
    # create optimizer
    params = list(model.parameters())
    #optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), eps=1e-9, 
    #    lr=args.learning_rate, weight_decay = 0.0001)
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=0.7
        , weight_decay=0.0001)
    learning_rate_scheduler = None
    learning_rate_scheduler = StepLR(optimizer, step_size = 60, gamma = 0.1)
    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce = False)
    model = model.to(device)
    #model = nn.DataParallel(model, device_ids=[0,1])

    print ("model created & starting training on", device, "...\n\n", )
    min_val_ls = 100000000000000000000
    no_improve = 0
    max_acc = 0
    # Training script
    for epoch in (range(args.num_epochs)):

        #model.train()
        train_ls , train_acc = train(model = model, epoch = epoch, mode = "Train", data_loader = data_loader, device = device,
            loss_func = cross_entropy_loss, optimizer = optimizer, scheduler = learning_rate_scheduler)

        model.eval()
        with torch.no_grad():
            val_ls, val_acc = train(model = model, epoch = epoch, mode = "Validation", data_loader = test_loader, device = device,
                loss_func = cross_entropy_loss, optimizer = optimizer, scheduler = learning_rate_scheduler)

        if val_ls <= min_val_ls  or max_acc < val_acc:
            no_improve = 0
            min_val_ls = val_ls
            max_acc = val_acc
            directory = "./saved_models/" + args.comment
            if os.path.exists(directory) is False:
                os.mkdir(directory)
            torch.save(model.state_dict(), directory + "/dl_project_" + str(epoch) + ".pth")
        else:
            no_improve +=1

        writer.add_scalars(f'',{
            'Acc_trn': train_acc,
            'Acc_val': val_acc,
            'Ls_trn': train_ls,
            'Ls_val': val_ls,}, epoch)

        if no_improve == args.patience:
            print("Early Stop")
            break

        
        if learning_rate_scheduler is not None:
            learning_rate_scheduler.step()
        
            

        print()
        # you can save the model here at specific epochs (ckpt) to load and evaluate the model on the val set

    print ()

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)
