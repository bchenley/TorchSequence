import torch 

class LRU(torch.nn.Module):
  '''
  Laguerre Recurrent Unit (LRU) model.

  Args:
      input_size (int): Number of expected features in the input.
      hidden_size (int): Number of features in the hidden state.
      weight_reg (list): Regularization parameters for the weights. [Regularization weight, regularization exponent]
      weight_norm (int): Norm to be used for weight regularization.
      bias (bool): If True, adds a learnable bias to the output.
      relax_init (list): Initial relaxation values for the LRU model.
      relax_train (bool): Whether to train the relaxation values.
      relax_minmax (list): Minimum and maximum relaxation values for each filter bank.
      device (str): Device to use for computation ('cpu' or 'cuda').
      dtype (torch.dtype): Data type of the model parameters.
  '''

  def __init__(self,
               input_size, hidden_size, 
               num_filterbanks = 1,
               weight_reg=[0.001, 1], weight_norm=2, 
               bias=False,
               relax_init=[0.5], relax_train=True, relax_minmax=[[0.1, 0.9]], 
               device = 'cpu', dtype = torch.float32):

    super(LRU, self).__init__()

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])
      
    # self.to(device = self.device, dtype = self.dtype)
    
    if len(relax_init) == 1: self.relax_init = self.relax_init * self.num_filterbanks

    if len(self.relax_minmax) == 1: self.relax_minmax = self.relax_minmax * self.num_filterbanks

    self.relax_init = torch.tensor(self.relax_init).reshape(self.num_filterbanks,)

    self.relax = torch.nn.Parameter(self.relax_init.to(device = self.device, dtype = self.dtype), requires_grad = self.relax_train)

    if self.input_size > 1:
      self.input_block = torch.nn.Linear(in_features = self.input_size, 
                                         out_features = self.num_filterbanks, bias = self.bias,
                                         device = self.device, dtype = self.dtype) 
    else:
      self.input_block = torch.nn.Identity()

  def init_hiddens(self, num_samples):
    '''
    Initialize the hidden state of the LRU model.

    Args:
        num_samples (int): Number of samples in the batch.

    Returns:
        torch.Tensor: Initialized hidden state tensor.
    '''
    return torch.zeros((self.num_filterbanks, num_samples, self.hidden_size)).to(device=self.device, dtype=self.dtype)

  def cell(self, input, hiddens=None):
    '''
    LRU cell computation for a single time step.

    Args:
        input (torch.Tensor): Input tensor for the current time step.
        hiddens (torch.Tensor): Hidden state tensor.

    Returns:
        torch.Tensor: Output tensor for the current time step.
        torch.Tensor: Updated hidden state tensor.
    '''
    num_samples, input_size = input.shape

    hiddens = hiddens if hiddens is not None else self.init_hiddens(num_samples)

    sq_relax = torch.sqrt(self.relax)

    hiddens_new = torch.zeros_like(hiddens).to(hiddens)

    hiddens_new[..., 0] = sq_relax[:, None] * hiddens[..., 0] + (1 - sq_relax ** 2).sqrt()[:, None] * self.input_block(input).t()

    for i in range(1, self.hidden_size):
        hiddens_new[..., i] = sq_relax[:, None] * (hiddens[..., i] + hiddens_new[..., i - 1]) - hiddens[..., i - 1]

    output = hiddens_new.permute(1, 0, 2)  # [batch_size, num_filters, hidden_size]

    return output, hiddens_new

  def forward(self, input, hiddens=None):
    '''
    Forward pass of the LRU model.

    Args:
        input (torch.Tensor): Input tensor.
        hiddens (torch.Tensor): Hidden state tensor.

    Returns:
        torch.Tensor: Output tensor.
        torch.Tensor: Updated hidden state tensor.
    '''
    num_samples, input_len, input_size = input.shape

    hiddens = self.init_hiddens(num_samples) if hiddens is None else hiddens

    output = []
    for n, input_n in enumerate(input.split(1, 1)):
        output_n, hiddens = self.cell(input_n.squeeze(1), hiddens)
        output.append(output_n.unsqueeze(1))

    output = torch.cat(output, 1)

    return output, hiddens

  def generate_laguerre_functions(self, max_len = None):
    '''
    Generate Laguerre functions up to a specified maximum length.

    Args:
        max_len (int): Maximum length of the Laguerre functions.

    Returns:
        torch.Tensor: Generated Laguerre functions.
    '''

    if max_len is None:
      max_len = ((-30 - torch.log(1-self.relax.max())) / torch.log(self.relax.max())).round().int()
    
    with torch.no_grad():
        hiddens = self.init_hiddens(1)

        impulse = torch.zeros((1, max_len, self.input_size)).to(device=self.device, dtype=self.dtype)

        impulse[:, 0, :] = 1

        output, hiddens = self.forward(impulse, hiddens)

        return output.squeeze(0)

  def clamp_relax(self):
    '''
    Clamp relaxation values to the specified minimum and maximum range.
    '''
    for i in range(self.num_filterbanks):
        self.relax[i].data.clamp_(self.relax_minmax[i][0], self.relax_minmax[i][1])
