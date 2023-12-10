from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
import pandas as pd
import cv2

def get_default_device():
  return torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

def to_device(entity,device):
  if isinstance(entity,(list,tuple)):
    return(to_device(elen,device) for elen in entity)

  return entity.to(device,non_blocking=True)

class Flower102(Dataset):
    def __init__(self,annotations_file,img_dir,transform = None,target_transform = None):
        # self.train = train
        self.img_labels =(annotations_file) #read csv
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)
    

    def __getitem__(self,idx):
        
        img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
        # import pdb; pdb.set_trace()
        # image = read_image(img_path)
        image = cv2.imread(img_path)
        image = torch.tensor(image,dtype=torch.float32)
        image = image.permute(2,0,1)
        # image.dtype=torch.float32
        # image = cv2.imread(img_path)
        # image = torch.from_numpy(image).permute(2,0,1)
        # image = image.numpy()
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label
# Function to add salt-and-pepper noise using OpenCV
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()

    # Salt noise
    salt_pixels = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt_pixels] = 1.0

    # Pepper noise
    pepper_pixels = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper_pixels] = 0.0

    return noisy_image
  
# Custom transform with salt-and-pepper noise and Gaussian blur
class SaltPepperNoiseAndBlurTransform:
    def __init__(self, salt_prob, pepper_prob, kernel_size=(5, 5)):
        self.salt_prob = salt_prob
        self.pepper_prob = pepper_prob
        self.kernel_size = kernel_size

    def __call__(self, img):
        # Convert PyTorch tensor to NumPy array
        img_np = img.numpy().transpose((1, 2, 0))

        # Add salt-and-pepper noise
        img_np = add_salt_and_pepper_noise(img_np, self.salt_prob, self.pepper_prob)

        # Gaussian blur
        img_np = cv2.GaussianBlur(img_np, self.kernel_size, 0)

        # Convert back to PyTorch tensor
        img_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float()

        return img_tensor

def get_default_device():
  return torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

def denormalize_image(image,mean,std):
   mean = torch.tensor(mean).reshape(1,3,1,1).to(device='cuda')
   std = torch.tensor(std).reshape(1,3,1,1).to(device='cuda')
   return (image*std + mean).squeeze(dim=0)
class DeviceDataLoader():
  """wrapper around dataloaders to transfer batch to specified devices"""
  def __init__(self,dataloader,device):
    self.dl = dataloader
    self.device = device
  def __iter__(self):
    for b in self.dl:
      yield to_device(b,self.device)
  def __len__(self):
    return len(self.dl)
  
device = get_default_device()
def make_dataloaders(train_dl,val_dl,test_dl):
    train_dl = DeviceDataLoader(train_dl,device)
    val_dl = DeviceDataLoader(val_dl,device)
    test_dl = DeviceDataLoader(test_dl,device)
    return train_dl,val_dl,test_dl

flower_names = [
    "alpine sea holly", "buttercup", "fire lily",
    "anthurium", "californian poppy", "foxglove",
    "artichoke", "camellia", "frangipani",
    "azalea", "canna lily", "fritillary",
    "ball moss", "canterbury bells", "garden phlox",
    "balloon flower", "cape flower", "gaura",
    "barbeton daisy", "carnation", "gazania",
    "bearded iris", "cautleya spicata", "geranium",
    "bee balm", "clematis", "giant white arum lily",
    "bird of paradise", "colt's foot", "globe thistle",
    "bishop of llandaff", "columbine", "globe-flower",
    "black-eyed susan", "common dandelion", "grape hyacinth",
    "blackberry lily", "corn poppy", "great masterwort",
    "blanket flower", "cyclamen", "hard-leaved pocket orchid",
    "bolero deep blue", "daffodil", "hibiscus",
    "bougainvillea", "desert-rose", "hippeastrum",
    "bromelia", "english marigold", "japanese anemone",
    "king protea", "peruvian lily", "stemless gentian",
    "lenten rose", "petunia", "sunflower",
    "lotus", "pincushion flower", "sweet pea",
    "love in the mist", "pink primrose", "sweet william",
    "magnolia", "pink-yellow dahlia?", "sword lily",
    "mallow", "poinsettia", "thorn apple",
    "marigold", "primula", "tiger lily",
    "mexican aster", "prince of wales feathers", "toad lily",
    "mexican petunia", "purple coneflower", "tree mallow",
    "monkshood", "red ginger", "tree poppy",
    "moon orchid", "rose", "trumpet creeper",
    "morning glory", "ruby-lipped cattleya", "wallflower",
    "orange dahlia", "siam tulip", "water lily",
    "osteospermum", "silverbush", "watercress",
    "oxeye daisy", "snapdragon", "wild pansy",
    "passion flower", "spear thistle", "windflower",
    "pelargonium", "spring crocus", "yellow iris"
]