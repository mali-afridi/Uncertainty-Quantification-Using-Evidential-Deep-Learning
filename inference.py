import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
from utils import flower_names,Flower102,SaltPepperNoiseAndBlurTransform,denormalize_image
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

test_transforms_flower_rigorous = transforms.Compose([
    transforms.Resize((224, 244)),
    SaltPepperNoiseAndBlurTransform(salt_prob=0.2, pepper_prob=0.2, kernel_size=(9, 9)),
    transforms.Normalize(*data_statistics, inplace=True)  # -1 to 1 (data = data-mean)/std
])

test_data_reference = Flower102(test_pd,images_path,transform=test_transforms_flower_reference)
test_data_rigorous = Flower102(test_pd,images_path,transform=test_transforms_flower_rigorous)

test_dataloader_reference = DataLoader(test_data_reference,batch_size=1,shuffle = False)
test_dataloader_rigorous = DataLoader(test_data_rigorous,batch_size=1,shuffle = False)


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

for (imagesr,labels1),(images_noise,labels2) in zip(test_dataloader_reference,test_dataloader_rigorous):
    imagesr,labels1 = imagesr.to(device='cuda'),labels1.to(device='cuda')
    images_noise = images_noise.to(device='cuda')

    logitsr = model(imagesr)
    # import pdb; pdb.set_trace()
    confidence_scores = torch.nn.functional.softmax(logitsr)
    conf = torch.max(confidence_scores).item()
    conf = round(conf,3)
    pred_class = torch.argmax(confidence_scores)
    image = denormalize_image(imagesr,*data_statistics)
    _,unc = get_edl_probs(logitsr)
    unc = round(unc.item(), 3)

    logits_noise  = model(images_noise)
    softmax = torch.nn.Softmax()
    confidence_scores_noise = softmax(logits_noise)
    conf_noise = torch.max(confidence_scores_noise)
    pred_class_noise = torch.argmax(confidence_scores_noise)
    image_noise = denormalize_image(images_noise,*data_statistics)
    # import pdb; pdb.set_trace()
    _,unc_noise = get_edl_probs(logits_noise)
    unc_noise = round(unc_noise.item(), 3)
    # fig, axs = plt.subplots(1, 2)
    # import pdb; pdb.set_trace()

    title = f"Pred Class: {pred_class.item()} true_label: {int(labels1.item())}"
    image_noise= image_noise.permute(1,2,0).detach().cpu().numpy()
    image= image.permute(1,2,0).detach().cpu().numpy()
    bottom = f"unc:{unc}"
    

    # Define the position and font settings

    text_position = (0, 10)
    image = cv2.putText(image, title, text_position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)
    # image = cv2.putText(image, f"Conf:{conf}", ((0,30)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)
    if(unc) > 0.5:
        color = (0, 0, 255)
    else:
        color = (0,255,0)
    image = cv2.putText(image, bottom, ((65,220)), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 1)
    bottom = f"unc:{unc_noise}"
    image_noise = cv2.putText(image_noise, title, text_position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)
    if(unc_noise) > 0.5:
        color = (0, 0, 255)
    else:
        color = (0,255,0)
    image_noise = cv2.putText(image_noise, bottom, ((65,220)), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color, 1)
    
    cv2.imwrite('noise.png', image_noise)
    
    cv2.imwrite('no_noise.png',image)
    # axs[0].imshow(image.permute(1,2,0).detach().cpu().numpy())
    # cv2.imshow("test",image.permute(1,2,0).detach().cpu().numpy())
    # cv2.title(f"Pred Class:{flower_names[pred_class]}, true_label{flower_names[int(labels1.item())]} with unc:{unc.item()}")
    # axs[0].set_title(f"Pred Class:{[pred_class.item()]}, true_label{[int(labels1.item())]} with unc:{unc.item()}")
    # axs[1].imshow(image_noise.permute(1,2,0).detach().cpu().numpy())
    # axs[1].set_title(f"Pred Class:{[pred_class_noise.item()]}, true_label{[int(labels1.item())]} with unc:{unc_noise.item()}")

    # # Adjust layout for better spacing
    # plt.tight_layout()

    # Show the plot
    # plt.savefig("comparison.png")
    # plt.show()
    # cv2.imwrite("test.png",image)
    # if (pred_class_noise.item()!=labels1.item()):
    import pdb; pdb.set_trace()

