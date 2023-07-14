import torch

class PositionalEncoding(torch.nn.Module):
  '''
  Positional encoding layer that adds positional information to the input.

  Args:
    dim (int): Dimensionality of the input.
    seq_len (int): Length of the input sequence.
    encoding_type (str, optional): Type of positional encoding to use. Supported types: 'absolute', 'relative'.
                                    Defaults to 'absolute'.
    device (str, optional): Device on which the positional encoding layer is allocated. Defaults to 'cpu'.
    dtype (torch.dtype, optional): Data type of the positional encoding layer. Defaults to torch.float32.
  '''

  def __init__(self,
                dim, seq_len, encoding_type='absolute',
                device='cpu', dtype=torch.float32):
      super(PositionalEncoding, self).__init__()

      locals_ = locals().copy()

      for arg in locals_:
        if arg != 'self':
          setattr(self, arg, locals_[arg])
        
      self.positional_encoding = self.generate_positional_encoding()

  def generate_positional_encoding(self):
      '''
      Generates the positional encoding based on the encoding type.

      Returns:
        torch.Tensor: Positional encoding tensor of shape (seq_len, dim).
      '''

      position = torch.arange(self.seq_len).unsqueeze(1).to(device=self.device, dtype=self.dtype)

      if self.encoding_type == 'absolute':
          positional_encoding = torch.zeros((self.seq_len, self.dim)).to(device=self.device, dtype=self.dtype)

          scaler = torch.exp(torch.arange(0, self.dim, 2) * -(torch.math.log(10000.0) / self.dim)).to(
              device=self.device, dtype=self.dtype)

          positional_encoding[:, 0::2] = torch.sin((position) * scaler)
          positional_encoding[:, 1::2] = torch.cos((position) * scaler)

      elif self.encoding_type == 'relative':
          positional_encoding = (position.repeat(1, self.dim) +
                                  torch.arange(self.dim).reshape(1, -1).to(device=self.device, dtype=self.dtype)) / self.seq_len

          positional_encoding = positional_encoding / positional_encoding.max()

      return positional_encoding

  def forward(self, input):
      '''
      Forward pass of the positional encoding layer.

      Args:
        input (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).

      Returns:
        torch.Tensor: Input tensor with added positional encoding of shape (batch_size, seq_len, dim).
      '''

      return input + self.positional_encoding[:input.shape[1], :]
