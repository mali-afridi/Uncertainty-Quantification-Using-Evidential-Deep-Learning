from training_script import evaluate
from training_script import train
from torchvision.models import resnet50, ResNet50_Weights
from losses.edl import EDLLoss
from main import make_data


import torch
import torch.nn as nn
# import pdb; pdb.set_trace()
#our dataloaders
train_dl,val_dl,test_dl = make_data()

print("Creating model")
#imagenet weight initialization
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#we have 102 classes
model.fc = nn.Linear(in_features=2048,out_features=102)

print(model)

model = model.to(device='cuda')

loss_func= nn.CrossEntropyLoss()

_,test_acc = evaluate(model,test_dl, loss_func)

print("Imagenet2k weights giving the accuracy on test data as:")
print(test_acc)


epochs =50
max_lr = 1e-2
print(f"Pretraining for epochs: {epochs} and learning rate {max_lr} with loss function CE")


loss_func= nn.CrossEntropyLoss()
optim = torch.optim.Adam

results = train(model,train_dl,val_dl,epochs,max_lr,loss_func,optim,'CE')
for result in results:
  print(result["avg_valid_acc"].item())


_,test_acc = evaluate(model,test_dl, nn.functional.cross_entropy)
print("Accuracy on the test set of flower 102 with ce loss:")
_,test_acc = evaluate(model,test_dl, loss_func)
print(test_acc)
torch.save(model.state_dict(),"ce.pth")


