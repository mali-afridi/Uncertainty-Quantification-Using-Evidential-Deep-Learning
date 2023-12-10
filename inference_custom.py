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


data_statistics = ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))

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
custom_images_path= '/home2/ali_afridi/FYP/cv_project/data/images/custom/'
result_images_path = '/home2/ali_afridi/FYP/cv_project/data/images/results/'

for images in os.listdir(custom_images_path):

    image = cv2.imread(custom_images_path+images) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

    # Convert the image to float32


    image_rgb = cv2.resize(image_rgb, (224,224), interpolation=cv2.INTER_LINEAR)
    image_rgb = image_rgb.astype(np.float32) 
    # Define mean and std for normalization
    mean = np.array(data_statistics[0])
    std = np.array(data_statistics[1])

    # Normalize the image
    normalized_image = (image_rgb - mean) / std



    image_rgb = torch.tensor(normalized_image,dtype=torch.float32)
    image_rgb = image_rgb.to(device='cuda')
    image_rgb = image_rgb.permute(2,0,1).unsqueeze(dim=0)

    logitsr = model(image_rgb)
    # import pdb; pdb.set_trace()
    confidence_scores = torch.nn.functional.softmax(logitsr)

    pred_class = torch.argmax(confidence_scores)
    image = denormalize_image(image_rgb,*data_statistics)
    _,unc = get_edl_probs(logitsr)
    unc = round(unc.item(), 3)




    title = f"Pred Class: {pred_class.item()}"
    # image = image.squeeze()
    # import pdb;pdb.set_trace()
    image= image.permute(1,2,0).detach().cpu().numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    bottom = f"unc:{unc}"


    # Define the position and font settings

    text_position = (0, 10)
    image = cv2.putText(image, title, text_position, cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)
    # image = cv2.putText(image, f"Conf:{conf}", ((0,30)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)
    if(unc) > 0.5:
        color = (0, 0, 255)
    else:
        color = (0,255,0)
    image = cv2.putText(image, bottom, ((50,220)), cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
    cv2.imwrite(result_images_path+images, image)

    # import pdb; pdb.set_trace()

