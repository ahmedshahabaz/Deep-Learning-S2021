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
def test(model, epoch, mode, data_loader, device, loss_func):
	
    total_correct_preds = 0.0
    loss = 0.0
    total_elements = 0
    output = [[]]* len(data_loader)
    with tqdm(data_loader, unit="batch") as tepoch:

        for step, (image_input, class_idxs, _ ) in (enumerate(tepoch)):

            tepoch.set_description(f"Epoch {epoch},{mode}")
            class_idxs = class_idxs.to(device)
            image_input = image_input.to(device)
            output[step] = model(image_input)

    return output

def main(args):
    
    args.mode = "train"

    test_loader, test_dataset = get_loader(args.data_dir, batch_size=args.batch_size, shuffle=False, 
                                           num_workers=args.num_workers, drop_last=False, args=args)

    data_size = test_dataset.get_data_size()
    num_classes = 11
    instance_size = test_dataset.get_instance_size()
    
    model1_1 = model1.my_ResNet(pretrained = False, num_classes = num_classes)
    model1_2 = model1.my_ResNet(pretrained = False, num_classes = num_classes)
    model1_3 = model1.my_ResNet(pretrained = False, num_classes = num_classes)
    model1_4 = model1.my_ResNet(pretrained = False, num_classes = num_classes)
    #model2 = models.my_ResNet(pretrained = False, num_classes = num_classes)    
    
    path1 = "./saved_weights/dl_project_"
    #path2 = "./final_models/dl_project_"
    
    #model1_1.load_state_dict(torch.load(path1 + str(83) + ".pth", map_location = device))
    #model1_2.load_state_dict(torch.load(path1 + str(85) + ".pth", map_location = device))
    #model1_3.load_state_dict(torch.load(path1 + str(92) + ".pth", map_location = device))
    model1_4.load_state_dict(torch.load(path1 + str(98) + ".pth", map_location = device))
    
    #model2.load_state_dict(torch.load(path1 + str(98) + ".pth", map_location = device))

    cross_entropy_loss = torch.nn.CrossEntropyLoss(reduce = False)
    final_output = []
    
    model_list =[model1_4] #[model1_1, model1_2, model1_3, model1_4]#, model2]

    for model in model_list:
        model.to(device)
        model.eval()
        with torch.no_grad():
            final_output += test(model = model, epoch = 0, mode = args.mode, data_loader = test_loader, device = device,
                loss_func = cross_entropy_loss)
    
    total_correct_preds = 0.0
    loss = 0.0
    total_elements = 0
    all_preds = torch.tensor([])
    #all_preds.to(device)
    all_targets = torch.tensor([])
    for step, data in enumerate(test_loader):
        
        output = final_output[step]
        class_idxs = data[1]
        class_idxs = class_idxs.to(device)
        state_loss = cross_entropy_loss(output, class_idxs)
        cur_loss = torch.sum(state_loss).item()
        total_elements += output.size(0)
        loss += cur_loss
        cur_loss = cur_loss/output.size(0)
        _, pred_idx = torch.max(F.softmax(output, dim = 1), dim = 1)
        all_preds = torch.cat( (all_preds, pred_idx.detach().cpu()), dim = 0)
        all_targets = torch.cat( (all_targets, class_idxs.detach().cpu()), dim = 0)

        correct_preds_batch = torch.sum(pred_idx==class_idxs).item()
        total_correct_preds += correct_preds_batch
    
    print()   
    print("Ac",round(total_correct_preds/total_elements,3),
        "Ls", round(loss/total_elements,3))


    from plotcm import plot_confusion_matrix
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(all_targets, all_preds)
    print(cm)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plot_confusion_matrix(cm, ["creamy_paste","diced","floured","grated","juiced","jullienne","mixed",
    "other","peeled","sliced","whole"])
    plt.show()
        # you can save the model here at specific epochs (ckpt) to load and evaluate the model on the val set

    print ()

if __name__ == '__main__':
    args = get_parser()
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    random.seed(1234)
    np.random.seed(1234)
    main(args)