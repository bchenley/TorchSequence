import torch 

class ModulationLayer(torch.nn.Module):
  '''
  Modulation layer that applies different modulation functions to the input.

  Args:
      window_len (int): Length of the input window.
      in_features (int): Number of input features.
      associated (bool): Whether the modulators are associated with each other.
      legendre_degree (int or None): Degree of the Legendre modulation function. If None, Legendre modulation is not applied.
      chebychev_degree (int or None): Degree of the Chebychev modulation function. If None, Chebychev modulation is not applied.
      dt (float): Time step for Fourier modulation.
      num_freqs (int or None): Number of frequencies for Fourier modulation. If None, Fourier modulation is not applied.
      freq_init (torch.Tensor or None): Initial frequencies for Fourier modulation. If None, frequencies are initialized uniformly.
      freq_train (bool): Whether to train the frequencies for Fourier modulation.
      phase_init (torch.Tensor or None): Initial phases for Fourier modulation. If None, phases are initialized as zeros.
      phase_train (bool): Whether to train the phases for Fourier modulation.
      num_sigmoids (int or None): Number of sigmoid functions for Sigmoid modulation. If None, Sigmoid modulation is not applied.
      slope_init (torch.Tensor or None): Initial slopes for Sigmoid modulation. If None, slopes are initialized from a normal distribution.
      slope_train (bool): Whether to train the slopes for Sigmoid modulation.
      shift_init (torch.Tensor or None): Initial shifts for Sigmoid modulation. If None, shifts are initialized from a uniform distribution.
      shift_train (bool): Whether to train the shifts for Sigmoid modulation.
      weight_reg (list): Regularization parameters for the linear function weights. [Regularization weight, regularization exponent]
      weight_norm (int): Norm to be used for weight regularization.
      zero_order (bool): Whether to include the zeroth-order term (constant) in the modulation functions.
      bias (bool): If True, adds a learnable bias to the linear function.
      pure (bool): If True, concatenates a constant term to the input.
      device (str): Device to use for computation ('cpu' or 'cuda').
      dtype (torch.dtype): Data type of the model parameters.

  '''

  def __init__(self, window_len, in_features, associated=False, legendre_degree=None, chebychev_degree=None,
               dt=1, num_freqs=None, freq_init=None, freq_train=True, phase_init=None, phase_train=True,
               num_sigmoids=None, slope_init=None, slope_train=True, shift_init=None, shift_train=True,
               weight_reg=[0.001, 1.], weight_norm=2, zero_order=True, bias=True, pure=False,
               device='cpu', dtype=torch.float32):

    super(ModulationLayer, self).__init__()

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])
        
    idx = 1

    self.num_modulators, m = 0, 0

    self.F = []

    self.modulators = torch.nn.ModuleList([])
    F_legendre, self.legendre_idx = None, None
    if self.legendre_degree is not None:
        m += 1
        F_legendre = LegendreModulator(window_len = self.window_len, scale = True, degree = self.legendre_degree,
                                       zero_order = self.zero_order, device = self.device, dtype = self.dtype)
        self.modulators.append(F_legendre)
        F.append(F_legendre.functions)
        self.legendre_idx = [m, torch.arange(idx, idx + F_legendre.num_modulators)]
        idx += F_legendre.num_modulators

    F_chebychev, self.chebychev_idx = None, None
    if self.chebychev_degree is not None:
        m += 1
        F_chebychev = ChebychevModulator(window_len = self.window_len, scale = True, kind = 1, degree = self.chebychev_degree,
                                         zero_order = self.zero_order * (len(self.F) == 0), device = self.device, dtype = self.dtype)
        self.modulators.append(F_chebychev)
        self.F.append(F_chebychev.functions)
        self.chebychev_idx = [m, torch.arange(idx, idx + F_chebychev.num_modulators)]
        idx += F_chebychev.num_modulators

    F_fourier, self.fourier_idx = None, None
    if self.num_freqs is not None:
        m += 1
        F_fourier = FourierModulator(window_len = self.window_len, num_freqs = self.num_freqs, dt = self.dt,
                                      freq_init = self.freq_init, freq_train = self.freq_train,
                                      phase_init = self.phase_init, phase_train = self.phase_train,
                                      device = self.device, dtype = self.dtype)
        self.modulators.append(F_fourier)
        self.F.append(F_fourier.functions)
        self.fourier_idx = [m, torch.arange(idx, idx + F_fourier.num_modulators)]
        idx += F_fourier.num_modulators

    F_sigmoid, self.sigmoid_idx = None, None
    if self.num_sigmoids is not None:
        m += 1
        F_sigmoid = SigmoidModulator(window_len = self.window_len, num_sigmoids = self.num_sigmoids,
                                      slope_init = self.slope_init, slope_train = self.slope_train,
                                      shift_init = self.shift_init, shift_train = self.shift_train,
                                      device = self.device, dtype = self.dtype)
        self.modulators.append(F_sigmoid)
        self.F.append(F_sigmoid.functions)
        self.sigmoid_idx = [m, torch.arange(idx, idx + F_sigmoid.num_modulators)]
        idx += F_sigmoid.num_modulators

    self.F = torch.cat(self.F, -1)

    self.num_modulators = self.F.shape[-1]

    self.linear_fn = HiddenLayer(in_features = self.in_features + int(self.pure),
                                 out_features = self.num_modulators,
                                 bias = self.bias,
                                 activation = 'identity',
                                 weight_reg = self.weight_reg,
                                 weight_norm = self.weight_norm,
                                 device = self.device, dtype = self.dtype)

  def forward(self, input, steps):
    '''
    Perform a forward pass through the modulation layer.

    Args:
        input (torch.Tensor): Input tensor.
        steps (int): Index of the modulation step.

    Returns:
        torch.Tensor: Output tensor.

    '''
    num_samples, seq_len, input_size = input.shape

    if self.pure:
      input_ = torch.cat((torch.ones((num_samples, seq_len, 1)).to(device=self.device, dtype=self.dtype), input), -1).to(input)
    else:
      input_ = input

    output = self.F[steps] * self.linear_fn(input_)

    return output

  def constrain(self):
    '''
    Apply constraints to the modulation parameters.

    '''
    if self.weight is not None:
      self.weight.data = self.weight.data / self.weight.data.norm(self.weight_norm, dim=1, keepdim=True)
      self.weight.data = self.weight.data.sum(dim=1, keepdim=True).sign() * self.weight.data

    if self.fourier_idx is not None:
      self.modulators[self.fourier_idx[0]].f = self.modulators[self.fourier_idx[0]].f.data.clamp_(0, 1 / (2 * self.dt))
      self.modulators[self.fourier_idx[0]].p = self.modulators[self.fourier_idx[0]].p.data.clamp_(-torch.pi, torch.pi)

  def penalize(self):
    '''
    Compute the regularization penalty.

    Returns:
      float: Regularization penalty.

    '''
    penalty = 0.
    if self.weight is not None:
        penalty += self.weight_reg[0] * torch.norm(self.weight, p=self.weight_reg[1]) * int(self.weight.requires_grad)

    return penalty
