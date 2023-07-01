import torch

class FourierModulator(torch.nn.Module):
  '''
  Fourier modulation function.

  Args:
    window_len (int): Length of the input window.
    num_freqs (int): Number of frequencies.
    dt (float): Time step.
    freq_init (torch.Tensor or None): Initial frequencies. If None, frequencies are initialized uniformly.
    freq_train (bool): Whether to train the frequencies.
    phase_init (torch.Tensor or None): Initial phases. If None, phases are initialized as zeros.
    phase_train (bool): Whether to train the phases.
    device (str): Device to use for computation ('cpu' or 'cuda').
    dtype (torch.dtype): Data type of the model parameters.

  '''

  def __init__(self,
               window_len, num_freqs, dt=1, freq_init=None, freq_train=True, phase_init=None,
               phase_train=True, device='cpu', dtype=torch.float32):
    super(FourierModulator, self).__init__()

    if freq_init is None:
        freq_init = ((1 / dt) / 4) * torch.ones(size=(1, num_freqs))
    else:
        freq_init = freq_init

    if phase_init is None:
        phase_init = torch.zeros(size=(1, num_freqs))
    else:
        phase_init = phase_init

    freq = torch.nn.Parameter(data=freq_init.to(device=device, dtype=dtype), requires_grad=freq_train)
    phase = torch.nn.Parameter(data=phase_init.to(device=device, dtype=dtype), requires_grad=phase_train)

    self.dt = dt
    self.window_len = window_len
    self.freq, self.phase = freq, phase
    self.num_modulators = num_freqs
    self.device, self.dtype = device, dtype

    self.generate_basis_functions()

  def generate_basis_functions(self):
    '''
    Generate the Fourier basis functions.

    Returns:
        torch.Tensor: Fourier basis functions.

    '''
    t = self.dt * torch.arange(0, self.window_len).view(-1, 1).to(device=self.device, dtype=self.dtype)

    y = torch.sin(2 * torch.pi * t * self.freq + self.phase)

    self.functions = y

    return y

  def forward(self, X, steps):
    '''
    Apply the Fourier modulation to the input.

    Args:
        X (torch.Tensor): Input tensor.
        steps (int): Index of the modulation step.

    Returns:
        torch.Tensor: Modulated tensor.

    '''
    X = X.to(device=self.device, dtype=self.dtype)

    self.functions = self.generate_basis_functions()

    y = X[:, :, None, :] * self.functions[steps]

    self.functions = y

    return y

