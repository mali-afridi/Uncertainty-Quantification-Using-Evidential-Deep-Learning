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

print("Making model2:")


model2 = resnet50()
model2.fc = nn.Linear(in_features=2048,out_features=102)

model2 = model2.to(device='cuda')
model2.load_state_dict(torch.load("ce.pth"))
print("Importing state dicts of ce model:")

loss_func= nn.CrossEntropyLoss()
_,test_acc = evaluate(model2,test_dl, loss_func)
print("Accuracy on the test set of flower 102:")
print(test_acc)

epochs =500
max_lr = 2e-2
print(f"New loss func (EDL) training: for {epochs} epochs and lr = {max_lr}")


loss_func2 = EDLLoss()
optim = torch.optim.SGD

results = train(model2,train_dl,val_dl,epochs,max_lr,loss_func2,optim,'EDL')
for result in results:
  print(result["avg_valid_acc"].item())


# _,test_acc = evaluate(model2,test_dl, nn.functional.cross_entropy)
print("Accuracy according to the edl loss on test set of flower 102: ")
_,test_acc = evaluate(model2,test_dl, loss_func2)
print(test_acc)
torch.save(model2.state_dict(),"unc500.pth")
