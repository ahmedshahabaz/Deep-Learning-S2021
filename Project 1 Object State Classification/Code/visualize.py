import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import cv2 as cv
import argparse
from torchvision import models, transforms

import model1 #, models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
    help='path to image')
args = vars(ap.parse_args())

#model2 = models.my_ResNet(pretrained = False, num_classes = 11)

#model1 = model1.my_ResNet(pretrained = False, num_classes = 11)


model1 = models.resnet50(pretrained=False)

path1 = "./saved_weights/dl_project_"
path2 = "./final_models/dl_project_"

#model1.load_state_dict(torch.load(path1 + str(98) + ".pth", map_location = device))

#model2.load_state_dict(torch.load(path2 + str(96) + ".pth", map_location = device))

model1_children = list(model1.children())

#model2_children = list(models.children())


model1_weights = [] # we will save the conv layer weights in this list
conv_layers = [] # w

#print(model1_children[0])

counter = 0 
# append all the conv layers and their respective weights to the list
for i in range(len(model1_children)):
    if type(model1_children[i]) == nn.Conv2d:
        counter += 1
        model1_weights.append(model1_children[i].weight)
        conv_layers.append(model1_children[i])
    elif type(model1_children[i]) == nn.Sequential:
        for j in range(len(model1_children[i])):
            for child in model1_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model1_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolutional layers: {counter}")

'''
for weight, conv in zip(model_weights, conv_layers):
    # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
    print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
'''

print(len(model1_weights))
'''
plt.figure(figsize=(20, 17))
for i, filter in enumerate(model1_weights[0]):
    plt.subplot(8, 8, i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
    plt.imshow(filter[0, :, :].detach(), cmap='gray')
    plt.axis('off')
    plt.savefig('./filter.png')
plt.show()
'''


img = cv.imread(f"./{args['image']}")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#plt.imshow(img)
#plt.show()
# define the transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])
img = np.array(img)
# apply the transforms
img = transform(img)
print(img.size())
# unsqueeze to add a batch dimension
img = img.unsqueeze(0)
print(img.size())


results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    # pass the result from the last layer to the next layer
    results.append(conv_layers[i](results[-1]))
# make a copy of the `results`
outputs = results


for num_layer in range(len(outputs)):
    plt.figure(figsize=(30, 30))
    layer_viz = outputs[num_layer][0, :, :, :]
    layer_viz = layer_viz.data
    print(layer_viz.size())
    for i, filter in enumerate(layer_viz):
        if i == 64: # we will visualize only 8x8 blocks from each layer
            break
        plt.subplot(8, 8, i + 1)
        plt.imshow(filter, cmap='gray')
        plt.axis("off")
    print(f"Saving layer {num_layer} feature maps...")
    plt.savefig(f"./Outputs/layer_{num_layer}.png")
    # plt.show()
    plt.close()
