import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils import flower_names,Flower102
import torchvision.transforms as transforms
from scipy.io import loadmat
import pandas as pd

# Read the image using cv2

scheme = loadmat('/home2/ali_afridi/FYP/cv_project/data/setid.mat')
images_path = '/home2/ali_afridi/FYP/cv_project/data/images/jpg/'


test_id = scheme['tstid'][0]

#we have to create annotation file
annots = loadmat('/home2/ali_afridi/FYP/cv_project/data/imagelabels.mat')
labels = annots['labels'][0]-1
# import pdb; pdb.set_trace()
labels_df = pd.DataFrame(labels)
images_name = []

for images in sorted(os.listdir(images_path)):
    images_name.append(images)
images_df= pd.DataFrame(images_name)
dataset_pd = pd.concat([images_df,labels_df],axis = 1)

test_pd = dataset_pd.iloc[test_id-1]

data_statistics = ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

test_transforms_flower_reference=transforms.Compose([
        transforms.Resize((224,244)),

        transforms.Normalize(*data_statistics,inplace=True) #-1 to 1 (data = data-mean)/std

])


test_data_reference = Flower102(test_pd,images_path,transform=test_transforms_flower_reference)

test_dataloader_reference = DataLoader(test_data_reference,batch_size=1,shuffle = False)


def get_edl_probs(y_preds): #To calculate mean uncertainty of 1 full fuse box image
    if type(y_preds)!=torch.Tensor:
        y_preds = torch.tensor(y_preds)
    evidence = torch.nn.functional.relu(y_preds)
    alpha = evidence+1
    alpha_sum  = torch.sum(alpha,dim=1,keepdim=True)
    probs = alpha/alpha_sum
    uncertainities = y_preds.shape[1]/alpha_sum    
    return probs,uncertainities  



model = resnet50()
model.fc = nn.Linear(in_features=2048,out_features=102)

model = model.to(device='cuda')
print("Importing unc model:")
model.load_state_dict(torch.load("unc900.pth"))
model = model.to(device='cuda')
model.eval()
# Normalize the image
true_labels,predicted_labels = [],[]
all_labels,all_preds = [],[]
for (imagesr,labels1) in test_dataloader_reference:
    all_labels.append(int(labels1.item()))
    imagesr,labels1 = imagesr.to(device='cuda'),labels1.to(device='cuda')

    logitsr = model(imagesr)
    # import pdb; pdb.set_trace()
    confidence_scores = torch.nn.functional.softmax(logitsr)
    conf = torch.max(confidence_scores).item()
    conf = round(conf,3)
    pred_class = torch.argmax(confidence_scores)
    
    _,unc = get_edl_probs(logitsr)
    unc = round(unc.item(), 3)
    all_preds.append(int(pred_class.item()))
    #uncertainty check
    if(unc) < 0.65:
        predicted_labels.append(int(pred_class.item()))
        true_labels.append(int(labels1.item()))
    else:
        print(f"Uncertain case found: Predicted class: {int(pred_class.item())} and True label: {int(labels1.item())} with unc score: {unc}")

conf_matrix = confusion_matrix(true_labels, predicted_labels)
# Plot confusion matrix using seaborn
# Split the confusion matrix into five subplots
num_subplots = 5
classes_per_subplot = 20
last_subplot_classes = 22

for i in range(num_subplots):
    start_class = i * classes_per_subplot
    end_class = (i + 1) * classes_per_subplot
    if i == num_subplots - 1:
        end_class = start_class + last_subplot_classes
    conf_matrix_subplot = conf_matrix[start_class:end_class, start_class:end_class]

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_subplot, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(start_class, end_class),
                yticklabels=range(start_class, end_class))

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - Classes {start_class} to {end_class - 1}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Save each subplot as a separate image file
    plt.savefig(f'unc_filter_confusion_matrix_subplot_{start_class}_to_{end_class - 1}.png')

    # Optionally, you can close the plot to free up resources
    plt.close()
#without unc filtering:
conf_matrix = confusion_matrix(all_labels, all_preds)

num_subplots = 5
classes_per_subplot = 20
last_subplot_classes = 22

for i in range(num_subplots):
    start_class = i * classes_per_subplot
    end_class = (i + 1) * classes_per_subplot
    if i == num_subplots - 1:
        end_class = start_class + last_subplot_classes
    conf_matrix_subplot = conf_matrix[start_class:end_class, start_class:end_class]

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_subplot, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(start_class, end_class),
                yticklabels=range(start_class, end_class))

    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - Classes {start_class} to {end_class - 1}')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Save each subplot as a separate image file
    plt.savefig(f'non_unc_filter_confusion_matrix_subplot_{start_class}_to_{end_class - 1}.png')

    # Optionally, you can close the plot to free up resources
    plt.close()



import pdb; pdb.set_trace()

