import torch

class Sigmoid(torch.nn.Module):

  def __init__(self, 
               in_features, 
               slope_init = None, slope_train = True, slope_reg=[0.001, 1],
               shift_init = None, shift_train = True, shift_reg=[0.001, 1],
               bias = True,
               device = 'cpu', dtype = torch.float32):
    
    super(Sigmoid, self).__init__()

    locals_ = locals().copy()
  
    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])
        
    if self.slope_init is None:
        self.slope_init = torch.nn.init.normal_(torch.empty((1, self.in_features)), mean = 0, std=1 / self.in_features)
    self.slope = torch.nn.Parameter(data = self.slope_init.to(device = self.device, dtype = self.dtype), requires_grad = self.slope_train)

    if self.shift_init is None:
        self.shift_init = torch.nn.init.uniform_(torch.empty((1, self.in_features)), a=-1, b=1)
    self.shift = torch.nn.Parameter(data = self.shift_init.to(device=device, dtype = self.dtype), requires_grad = self.shift_train)

    if self.bias:
      self.bias = torch.nn.Parameter(data = torch.zeros(self.in_features,).to(device = self.device, dtype = self.dtype), requires_grad = True)
    else:
      self.bias = None

  def forward(self, X):

    y = 1 / (1 + torch.exp(-self.slope * (X - self.shift))) + self.bias

    return y
  
  def penalize(self):

    penalty = self.slope_reg[0] * torch.norm(self.slope, p=self.slope_reg[1]) * int(self.slope.requires_grad)
    penalty += self.shift_reg[0] * torch.norm(self.shift, p=self.shift_reg[1]) * int(self.shift.requires_grad)
    
    return penalty
