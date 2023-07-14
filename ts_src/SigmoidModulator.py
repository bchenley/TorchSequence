import torch

class SigmoidModulator(torch.nn.Module):
  '''
  Sigmoid modulation function.

  Args:
      window_len (int): Length of the input window.
      num_sigmoids (int): Number of sigmoid functions.
      scale (bool): If True, scale the input to the range [0, 1].
      slope_init (torch.Tensor or None): Initial slopes. If None, slopes are initialized from a normal distribution.
      slope_train (bool): Whether to train the slopes.
      shift_init (torch.Tensor or None): Initial shifts. If None, shifts are initialized from a uniform distribution.
      shift_train (bool): Whether to train the shifts.
      device (str): Device to use for computation ('cpu' or 'cuda').
      dtype (torch.dtype): Data type of the model parameters.

  '''

  def __init__(self, window_len, num_sigmoids, scale=True, slope_init=None, slope_train=True,
                shift_init=None, shift_train=True, device='cpu', dtype=torch.float32):
    super(SigmoidModulator, self).__init__()

    locals_ = locals().copy()
  
    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])
        
    if self.slope_init is None:
        self.slope_init = torch.nn.init.normal_(torch.empty((1, self.num_sigmoids)), mean = 0, std=1 / self.window_len)
    self.slope = torch.nn.Parameter(data = self.slope_init.to(device = self.device, dtype = self.dtype), requires_grad = self.slope_train)

    if self.shift_init is None:
        self.shift_init = torch.nn.init.uniform_(torch.empty((1, self.num_sigmoids)), a=-1, b=1)
    self.shift = torch.nn.Parameter(data = self.shift_init.to(device=device, dtype = self.dtype), requires_grad = self.shift_train)

    self.num_modulators = num_sigmoids
    
    self.functions = self.generate_basis_functions()

  def generate_basis_functions(self):
    '''
    Generate the sigmoid basis functions.

    Returns:
        torch.Tensor: Sigmoid basis functions.

    '''
    t = torch.arange(0, self.window_len).view(-1, 1).to(device=self.device, dtype=self.dtype)

    scaler = (t.max() - t.min()) if self.scale else 1

    y = 1 / (1 + torch.exp(-self.slope * (t - self.shift * scaler)))

    self.functions = y

    return y

  def forward(self, X, steps):
    '''
    Apply the sigmoid modulation to the input.

    Args:
        X (torch.Tensor): Input tensor.
        steps (int): Index of the modulation step.

    Returns:
        torch.Tensor: Modulated tensor.

    '''
    X = X.to(device=self.device, dtype=self.dtype)

    self.functions = self.generate_basis_functions()

    print(X.shape)
    print(self.functions[steps].shape)

    y = X[:, :, None, :] * self.functions[steps]

    return y
