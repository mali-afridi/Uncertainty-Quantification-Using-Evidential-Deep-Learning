import torch
import torch.nn as nn

def to_device(entity,device):
  if isinstance(entity,(list,tuple)):
    return(to_device(elen,device) for elen in entity)

  return entity.to(device,non_blocking=True)

#Uncertainty Training

softmax = nn.Softmax()
def accuracy(logits,labels):
  prediction_softmax = softmax(logits)
  pred, predClassId = torch.max(prediction_softmax,dim=1)
  return torch.tensor(torch.sum(predClassId==labels).item()/len(logits))

def evaluate(model,dl,loss_func):
 
   model.eval()
   batch_losses, batch_acc = [], []
   
   for images, labels in dl:
      images, labels = images.to(device='cuda'), labels.to(device='cuda')
      with torch.no_grad():

        logits = model(images)
      
      batch_losses.append(loss_func(logits,labels))
      batch_acc.append(accuracy(logits,labels))
  #  import pdb; pdb.set_trace()
   epoch_avg_loss = torch.stack(batch_losses).mean().item()
   epoch_avg_acc = torch.stack(batch_acc).mean()
   return epoch_avg_loss,epoch_avg_acc

def train(model,train_dl,val_dl,epochs,max_lr,loss_func,optim,b):
  name_loss = b
  name_functions = ["CE","EDL"]

  #initialize the optimizer
  optimizer = optim(model.parameters(),max_lr)
  # schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr,epochs*len(train_dl))

  results = []
  for epoch in range(epochs):
    model.train()
    train_losses = []
    # lrs = []

    # for images, labels in train_dl:
    #     images, labels = images.to(device='cuda'), labels.to(device='cuda')

    #     # Zero the gradients
    #     optimizer.zero_grad()

    #     # Forward pass
    #     logits = model(images)
    #     loss = loss_func(logits, labels)
    #     count+=1
    #     print(loss.item(),count)


    #     # Backward pass and optimization
    #     loss.backward()
    #     optimizer.step()

    # # Validation loop
    # # import pdb;pdb.set_trace()
    # model.eval()
    # with torch.no_grad():
    #     total_correct = 0
    #     total_samples = 0

    #     for val_inputs, val_labels in val_dl:
    #         val_inputs, val_labels = val_inputs.to(device='cuda'), val_labels.to(device='cuda')

    #         val_logits = model(val_inputs)
    #         _, predicted = torch.max(val_logits, 1)

    #         total_samples += val_labels.size(0)
    #         total_correct += (predicted == val_labels).sum().item()

    #     val_accuracy = total_correct / total_samples

    #     # Save the model if the validation accuracy is better than the previous best
    #     if val_accuracy > best_val_acc:
    #         best_val_acc = val_accuracy
    #         # best_model_weights = model.state_dict()

    #     # Print epoch statistics
    #     print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}")

    for images, labels in train_dl:
      images, labels = images.to(device='cuda'), labels.to(device='cuda')
      logits = model(images)
      #logits size = (batch,number of classes)
      #probs,uncertainties = get_edl_probs(logits) #which will be changed to softplus
      # import pdb; pdb.set_trace()
      if (name_loss == name_functions[0]):
      #test = torch.nn.functional.softmax(logits)
        # loss = loss_func((logits),labels)
        loss = loss_func((logits),labels)
      else:
        loss = loss_func(logits,labels)
      # import pdb;pdb.set_trace()
      train_losses.append(loss)
      loss.backward() #derivative of this loss wrt all network params
      optimizer.step() #take step towards direction of gradient descent
      optimizer.zero_grad()
      # lrs.append(optimizer.param_groups[0]["lr"])
      # schedular.step()
    epoch_train_loss = torch.stack(train_losses).mean().item()
    print(f"epoch number : {epoch+1} loss :{epoch_train_loss}")

    # import pdb;pdb.set_trace()
    epoch_avg_loss,epoch_avg_acc = evaluate(model,val_dl, loss_func)
    results.append({'avg_valid_loss': epoch_avg_loss, 'avg_valid_acc': epoch_avg_acc, 'avg_train_loss':epoch_train_loss})
    # results.append({'avg_valid_loss': epoch_avg_loss, 'avg_valid_acc': epoch_avg_acc, 'avg_train_loss':epoch_train_loss, 'lr': lrs})

  return results
