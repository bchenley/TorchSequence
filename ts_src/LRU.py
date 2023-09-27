import torch

from ts_src.HiddenLayer import HiddenLayer

class LRU(torch.nn.Module):
    
    """
    Linear Recurrent Unit (LRU) module.

    Args:
        input_size (int): Input feature size.
        hidden_size (int): Hidden state size.
        num_filterbanks (int, optional): Number of filterbanks. Default is 1.
        weight_reg (list of float, optional): Weight regularization for L2 regularization and Laplace prior.
        weight_norm (float, optional): Weight normalization. Default is 2.
        bias (bool, optional): Whether to use bias. Default is False.
        relax_init (list of float, optional): Relaxation factor initialization.
        relax_train (bool, optional): Whether relaxation factors are trainable. Default is True.
        relax_minmax (list of list of float, optional): Min and max values for relaxation factors.
        device (str, optional): Device to use. Default is 'cpu'.
        dtype (torch.dtype, optional): Data type to use. Default is torch.float32.
    """
    
    def __init__(self,
                 input_size, hidden_size,
                 num_filterbanks=1,
                 weight_reg=[0.001, 1], weight_norm=2,
                 bias=False,
                 relax_init=[0.5], relax_train=True, relax_minmax=[[0.1, 0.9]],
                 feature_associated=True,
                 input_block_weight_to_ones=False,
                 device='cpu', dtype=torch.float32):

        super(LRU, self).__init__()

        locals_ = locals().copy()

        for arg in locals_:
          if arg != 'self':
            setattr(self, arg, locals_[arg])

        if len(relax_init) == 1:
            self.relax_init = self.relax_init * self.num_filterbanks

        if len(self.relax_minmax) == 1:
            self.relax_minmax = self.relax_minmax * self.num_filterbanks

        self.relax_init = torch.tensor(self.relax_init).reshape(self.num_filterbanks,)

        self.relax = torch.nn.Parameter(
            self.relax_init.to(device=self.device, dtype=self.dtype), requires_grad=self.relax_train)

        if (self.input_size > 1) & (not self.feature_associated):
            self.input_block = HiddenLayer(in_features = self.input_size, 
                                           out_features = self.num_filterbanks,
                                           bias = self.bias,
                                           device = self.device, 
                                           weight_to_ones = self.input_block_weight_to_ones,
                                           dtype = self.dtype)

        else:
            self.input_block = torch.nn.Identity()

    def init_hiddens(self, num_samples):
        """
        Initialize hidden states.

        Args:
            num_samples (int): Number of samples.

        Returns:
            torch.Tensor: Initialized hidden states.
        """
        return torch.zeros((self.num_filterbanks, num_samples, self.hidden_size)).to(device=self.device, dtype=self.dtype)

    def cell(self, input, hiddens):
      """
      LRU cell operation.

      Args:
          input (torch.Tensor): Input tensor.
          hiddens (torch.Tensor, optional): Hidden states. Default is None.

      Returns:
          torch.Tensor: Output tensor.
          torch.Tensor: Updated hidden states.
      """
      num_samples, input_size = input.shape

      sq_relax = torch.sqrt(self.relax)

      hiddens_new = torch.zeros_like(hiddens).to(hiddens)
      
      hiddens_new[..., 0] = sq_relax[:, None] * hiddens[..., 0] + (1 - sq_relax ** 2).sqrt()[:, None] * self.input_block(input).t()

      for i in range(1, self.hidden_size):
        hiddens_new[..., i] = sq_relax[:, None] * (hiddens[..., i] + hiddens_new[..., i - 1]) - hiddens[..., i - 1]

      output = hiddens_new.permute(1, 0, 2)
      
      return output, hiddens_new

    def forward(self, input, hiddens=None):
        """
        LRU forward pass.

        Args:
            input (torch.Tensor): Input tensor.
            hiddens (torch.Tensor, optional): Hidden states. Default is None.

        Returns:
            torch.Tensor: Output tensor.
            torch.Tensor: Updated hidden states.
        """
        num_samples, input_len, input_size = input.shape
        
        if self.feature_associated & (self.num_filterbanks != input_size):
          raise ValueError(f"LRU is feature-associated, but the number of filterbanks ({self.num_filterbanks}) does not equal the number of input features ({input_size}).")

        hiddens = self.init_hiddens(num_samples) if hiddens is None else hiddens

        output = []
        for n, input_n in enumerate(input.split(1, 1)):
            output_n, hiddens = self.cell(input_n.squeeze(1), hiddens)
            output.append(output_n.unsqueeze(1))

        output = torch.cat(output, 1)

        return output, hiddens

    def generate_laguerre_functions(self, max_len=None):
        """
        Generate Laguerre functions.

        Args:
            max_len (int, optional): Maximum length. Default is None.

        Returns:
            torch.Tensor: Generated Laguerre functions.
        """
        if max_len is None:
            max_len = ((-30 - torch.log(1 - self.relax.max())) / torch.log(self.relax.max())).round().int()

        with torch.no_grad():
            hiddens = self.init_hiddens(1)

            impulse = torch.zeros((1, max_len, self.input_size)).to(device=self.device, dtype=self.dtype)

            impulse[:, 0, :] = 1

            output, hiddens = self.forward(impulse, hiddens)

            return output.squeeze(0)

    def clamp_relax(self):
      """Clamp relaxation factors within specified range."""
      for i in range(self.num_filterbanks):
          self.relax[i].data.clamp(self.relax_minmax[i][0], self.relax_minmax[i][1])
