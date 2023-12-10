import torch 
import torch.nn as nn
from torchvision.models import resnet50
from losses.edl import EDLLoss
from training_script_resume import train,evaluate
from main import make_data

model2 = resnet50()
model2.fc = nn.Linear(in_features=2048,out_features=102)
train_dl,val_dl,test_dl = make_data()
model2 = model2.to(device='cuda')
# model2.load_state_dict(torch.load("unc2.pth"))
# Load model checkpoint
epochs = 700
max_lr = 1e-2
print(f"resuming training from epoch number: {epochs} and lr {max_lr}")
checkpoint = torch.load("unc700.pth")
model2.load_state_dict(checkpoint)
# import pdb; pdb.set_trace()
# epochs = checkpoint['epoch']

total_epochs =900

# Resume training from a specific epoch
print(f"New loss func (EDL) training: for {total_epochs} epochs and lr = {max_lr}")


loss_func2 = EDLLoss()
optim = torch.optim.SGD

results = train(model2,train_dl,val_dl,epochs,total_epochs,max_lr,loss_func2,optim,'EDL')
for result in results:
  print(result["avg_valid_acc"].item())


# _,test_acc = evaluate(model2,test_dl, nn.functional.cross_entropy)
print("Accuracy according to the edl loss on test set of flower 102: ")
_,test_acc = evaluate(model2,test_dl, loss_func2)
print(test_acc)
torch.save(model2.state_dict(),"unc900.pth")