import torch 

class LRU(torch.nn.RNN):
  '''
  Laguerre Recurrent Unit (LRU) model based on RNN architecture.

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
               input_size, hidden_size, weight_reg=[0.001, 1], weight_norm=2, bias=False,
               relax_init=[0.5], relax_train=True, relax_minmax=[[0.1, 0.9]], device='cpu', dtype=torch.float32):

    super(LRU, self).__init__(input_size=input_size, hidden_size=hidden_size, batch_first=True)

    self.to(device=device, dtype=dtype)

    num_filterbanks = len(relax_init)

    if len(relax_minmax) == 1:
        relax_minmax = relax_minmax * num_filterbanks

    relax_init = torch.tensor(relax_init).reshape(num_filterbanks,)

    relax = torch.nn.Parameter(relax_init.to(device=device, dtype=dtype), requires_grad=relax_train)

    if input_size > 1:
        input_block = torch.nn.Linear(in_features=input_size, out_features=num_filterbanks, bias=bias)
    else:
        input_block = torch.nn.Identity()

    self.bias_hh_l0.requires_grad = False
    self.bias_hh_l0.requires_grad = False

    self.input_size, self.hidden_size = input_size, hidden_size
    self.num_filterbanks = num_filterbanks
    self.input_block = input_block
    self.relax_minmax = relax_minmax
    self.relax = relax
    self.weight_reg, self.weight_norm = weight_reg, weight_norm
    self.device, self.dtype = device, dtype

    # Remove built-in weights and biases
    self.weight_ih_l0 = None
    self.weight_hh_l0 = None
    self.bias_ih_l0 = None
    self.bias_hh_l0 = None

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

  def generate_laguerre_functions(self, max_len):
    '''
    Generate Laguerre functions up to a specified maximum length.

    Args:
        max_len (int): Maximum length of the Laguerre functions.

    Returns:
        torch.Tensor: Generated Laguerre functions.
    '''
    with torch.no_grad():
        hiddens = self.init_hiddens(1)

        impulse = torch.zeros((1, max_len, self.input_size)).to(device=self.device, dtype=self.dtype)

        impulse[:, 0, :] = 1

        output, hiddens = self.forward(impulse, hiddens)

        return output

  def clamp_relax(self):
    '''
    Clamp relaxation values to the specified minimum and maximum range.
    '''
    for i in range(self.num_filterbanks):
        self.relax[i].data.clamp_(self.relax_minmax[i][0], self.relax_minmax[i][1])