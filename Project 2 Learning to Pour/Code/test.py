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
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = None

def train(model, epoch, mode, data_loader, loss_func, optimizer, scheduler, train_mean, train_std):
    
    total_loss = 0.0
    total_item = 0

    with tqdm(data_loader, unit="batch") as tepoch:

        for step, (data, label, lenghts, nonZeroRows) in (enumerate(tepoch)):
            
            tepoch.set_description(f"Epoch {epoch},{mode}")
            
            normalize = transforms.Compose([transforms.Normalize(train_mean, train_std)])
            data = normalize(data.permute(2,0,1)).permute(1,2,0)

            data = data.to(device)
            label = label.to(device)
            
            output, padded_out = model(data, lenghts, mode = 'test')

            if step != 0:
                stacked_padded_out = torch.cat((stacked_padded_out ,padded_out), dim = 0)
            else:
                stacked_padded_out = padded_out

            loss = loss_func(output[nonZeroRows], label[nonZeroRows])

            total_loss += loss.item() * output[nonZeroRows].shape[0]
            total_item += output[nonZeroRows].shape[0]
            tepoch.set_postfix({"Lss":loss.item()})

        np.save("out_" +str(epoch) + ".npy", stacked_padded_out.cpu())
        mode = mode + "_"+str(epoch)
        print(mode , " loss: ", np.sqrt(total_loss/total_item))
        
        return np.sqrt(total_loss / total_item)

def main(args):
    
    global writer
    global device

    args.mode = "test"
    
    full_dataset,test_loader, train_mean, train_std = get_loader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
        drop_last=False, args=args)

    
    if torch.cuda.is_available():
        device = torch.device("cuda:"+ str(args.device))

    '''
    creating different models for Snapshot Ensemble
    '''

    args.lstm_layers = 1
    model1 = models.myRNN(args)
    
    args.lstm_layers = 2
    model2 = models.myRNN(args)

    args.lstm_layers = 4
    model3 = models.myRNN(args)
    
    '''
    corresponding model weights for Snapshot Ensemble
    '''

    weights = ['./model_weight/dl_project_1.pth' , './model_weight/dl_project_2.pth']#, './model_weight/dl_project_4.pth']

    #model1.load_state_dict(torch.load(weights[0], map_location = device))
    #model2.load_state_dict(torch.load(weights[1], map_location = device))
    #model3.load_state_dict(torch.load(weights[2], map_location = device))

    all_models = [model1, model2]#, model3]

    mse_loss = torch.nn.MSELoss(reduction = 'mean')
    test_loss = 0.0
    epoch = 0

    for model, weight in zip(all_models, weights):
        
        model.load_state_dict(torch.load(weight, map_location = device))
        model = model.to(device)
        model.eval()
        with torch.no_grad():

            test_loss += train(model = model, epoch = epoch, mode = "test", data_loader = test_loader,
                loss_func = mse_loss, optimizer = None, scheduler = None,
                train_mean = train_mean, train_std = train_std)

        epoch +=1
        print ()

    #print("Average Test Loss : ", test_loss/len(all_models))

    output = np.zeros((len(test_loader.dataset), 700, 1))

    for i in range(len(weights)):
        output += np.load("out_" + str(i) + ".npy")

    output = np.true_divide(output, len(weights))

    np.save("shahabaz_out.npy", output)

    print(output.shape)

    true_label = np.load(os.path.join("./data",args.npy_file))
    prediction = np.load('shahabaz_out.npy')
    #prediction = output

    '''
    Calculating loss from the saved output
    '''
    
    tot = 0
    cnt = 0
    for i in range(len(prediction)):
        for j in range(len(prediction[0])):
            if prediction[i][j][0]!=0:
                tot += (prediction[i][j][0] - true_label[i][j][1]) ** 2
                cnt += 1

    print("Average loss: ",(tot / cnt)**0.5)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("file_name")
    args = get_parser(parser)
    args.npy_file = args.file_name

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)