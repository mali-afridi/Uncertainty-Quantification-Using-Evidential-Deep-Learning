from utils import Flower102

from torchvision.models import resnet50
from torch.utils.data import Dataset, DataLoader

from scipy.io import loadmat
import os

from os import path
import pandas as pd
import torchvision.transforms as transforms
from utils import to_device,get_default_device


def make_data():
    scheme = loadmat('/home2/ali_afridi/FYP/cv_project/data/setid.mat')
    images_path = '/home2/ali_afridi/FYP/cv_project/data/images/jpg/'
    train_id = scheme['trnid'][0]
    val_id = scheme['valid'][0]
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


    train_pd = dataset_pd.iloc[train_id-1]
    test_pd = dataset_pd.iloc[test_id-1]
    val_pd = dataset_pd.iloc[val_id-1]

    # import pdb; pdb.set_trace()

    #we need transformation now
    data_statistics = ((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    train_transforms_flower = transforms.Compose([
            transforms.Resize((224,244)),
            transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(), #CxHxW

            transforms.Normalize(*data_statistics,inplace=True), #-1 to 1 (data = data-mean)/std
        ])

    test_transforms_flower=transforms.Compose([
            transforms.Resize((224,244)),
            #torch.RandomCrop(32,padding=4,padding_mode='reflect'),
            #torch.RandomHorizontalFlip(),
            # transforms.ToTensor(), #CxHxW
            transforms.Normalize(*data_statistics,inplace=True) #-1 to 1 (data = data-mean)/std

        ])

    training_data = Flower102(train_pd,images_path,transform=train_transforms_flower)
    val_data = Flower102(val_pd,images_path,transform=train_transforms_flower)
    test_data = Flower102(test_pd,images_path,transform=test_transforms_flower)
    # import pdb; pdb.set_trace()

    train_dataloader = DataLoader(training_data,batch_size=16,shuffle = True)

    val_dataloader = DataLoader(val_data,batch_size=16,shuffle = True)
    test_dataloader = DataLoader(test_data,batch_size=16,shuffle = True)

    return train_dataloader,val_dataloader,test_dataloader

