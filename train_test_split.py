import torch
import cv2
from torchvision.models import resnet50
import numpy as np
from scipy.io import loadmat
import os
import shutil
from os import path

# annots = loadmat('/home2/ali_afridi/FYP/cv_project/data/imagelabels.mat')
scheme = loadmat('/home2/ali_afridi/FYP/cv_project/data/setid.mat')
train_id = scheme['trnid'][0]
val_id = scheme['valid'][0]
test_id = scheme['tstid'][0]

images_path = '/home2/ali_afridi/FYP/cv_project/data/images/moving/jpg/'
train_path = '/home2/ali_afridi/FYP/cv_project/data/train/'
test_path = '/home2/ali_afridi/FYP/cv_project/data/test/'
val_path = '/home2/ali_afridi/FYP/cv_project/data/val/'
os.makedirs(train_path,exist_ok=True)
os.makedirs(test_path,exist_ok=True)
os.makedirs(val_path,exist_ok=True)




# import pdb; pdb.set_trace()
#training split
train_image_count=0
val_image_count=0
test_image_count=0
for train_images in train_id:
    moving = images_path+'image_0'+str(train_images).zfill(4)+'.jpg'
    if(path.exists(moving)):
        print('train_image found')
        # import pdb; pdb.set_trace()
        shutil.move(moving,train_path)
        train_image_count+=1
    else:
        print(f'train image not found for: {moving}')


for val_images in val_id:
    moving = images_path+'image_0'+str(val_images).zfill(4)+'.jpg'
    if(path.exists(moving)):
        print('val_image found')
        shutil.move(moving,val_path)
        val_image_count+=1
    else:
        print(f'val image not found for: {moving}')

for test_images in test_id:
    moving = images_path+'image_0'+str(test_images).zfill(4)+'.jpg'
    if(path.exists(moving)):
        print('test_image found')
        shutil.move(moving,test_path)
        test_image_count+=1
    else:
        print(f'test image not found for: {moving}')  
        
# print(np.unique(annots['labels'][0]))
print("Total images moved:")
print(train_image_count+ test_image_count+val_image_count)
print("Total images in dataset were:")
print(len(train_id)+len(test_id)+len(val_id))
# import pdb; pdb.set_trace()
# print(scheme)

