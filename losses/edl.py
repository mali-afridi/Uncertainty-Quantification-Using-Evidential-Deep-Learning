import torch
relu = torch.nn.ReLU()

#pass logits into a nonlinear function to get evidences
class EDLLoss(torch.nn.Module):
  def __init__(self,annealing_step = 0,num_classes = 102):
    super().__init__()
    self.num_classes = num_classes
    self.annealing_step = annealing_step

  def _get_evidence(self,y):
    # return torch.nn.functional.softplus(y)
      return relu(y)
  def _kl_divergence(self,alpha):
    device = alpha.device
    ones = torch.ones([1,self.num_classes],dtype = torch.float32,device = device)
    alpha_sum = torch.sum(alpha,dim=1,keepdim=True)
    first_term = (
        torch.lgamma(alpha_sum)
        - torch.lgamma(alpha).sum(dim=1,keepdim=True)
        +torch.lgamma(ones).sum(dim=1,keepdim=True)
        -torch.lgamma(ones.sum(dim=1,keepdim=True))
    )
    second_term = (
        (alpha-ones)
        .mul(torch.digamma(alpha)-torch.digamma(alpha_sum))
        .sum(dim=1,keepdim=True)
    )
    kl = first_term + second_term

    return kl
  def _loglikelihood_loss(self,y,alpha):
    device = alpha.device
    y= y.to(device)
    alpha = alpha.to(device)
    alpha_sum = torch.sum(alpha,dim =1,keepdim=True)
    loglikelihood_err = torch.sum((y-(alpha/alpha_sum))**2,dim=1,keepdim=True)
    loglikelihood_var = torch.sum(alpha*(alpha_sum-alpha)/(alpha_sum*alpha_sum*(alpha_sum+1)),dim=1,keepdim=True)
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood
  def __call__(self,output,target,epoch_num=None):
    evidence = self._get_evidence(output)
    alpha = evidence+1
    device = alpha.device
    if target.ndim ==1:
      y = torch.nn.functional.one_hot(target,num_classes = self.num_classes).float().to(device)
    else:
      y = target.to(device)
    alpha = alpha.to(device)
    loglikelihood = self._loglikelihood_loss(y,alpha)
    if self.annealing_step!=0:
      assert epoch_num is not None, "epoch_num must be provided when using annealing"
      annealing_coef = torch.min(
          torch.tensor(1.0,dtype = torch.float32),
          torch.tensor(epoch_num/self.annealing_step,dtype = torch.float32),
)
    else:
      annealing_coef = 1.0
    kl_alpha = (alpha - 1)*(1-y)+1
    kl_div = annealing_coef*self._kl_divergence(kl_alpha)
    loss = (loglikelihood+ kl_div).mean()
    return loss

