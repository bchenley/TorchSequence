import torch

class HiddenLayer(torch.nn.Module):
  '''
  Hidden layer module with various activation functions and regularization options.

  Args:
      in_features (int): Number of input features.
      out_features (int or None): Number of output features. If None or 0, output features will be the same as input features.
      bias (bool): If True, adds a learnable bias to the output.
      activation (str): Activation function to use. Options: 'identity', 'polynomial', 'tanh', 'sigmoid', 'softmax', 'relu'.
      weight_reg (list): Regularization parameters for the weights. [Regularization weight, regularization exponent]
      weight_norm (int): Norm to be used for weight regularization.
      degree (int): Degree of the polynomial activation function.
      coef_init (torch.Tensor): Initial coefficients for the polynomial activation function. If None, coefficients are initialized randomly.
      coef_train (bool): Whether to train the coefficients.
      coef_reg (list): Regularization parameters for the coefficients. [Regularization weight, regularization exponent]
      zero_order (bool): Whether to include the zeroth-order term (constant) in the polynomial activation function.
      softmax_dim (int): Dimension along which to apply softmax activation.
      dropout_p (float): Dropout probability.
      device (str): Device to use for computation ('cpu' or 'cuda').
      dtype (torch.dtype): Data type of the model parameters.

  '''

  def __init__(self, in_features, out_features=None, bias=True, activation='identity',
                weight_reg=[0.001, 1], weight_norm=2, degree=1, coef_init=None, coef_train=True,
                coef_reg=[0.001, 1], zero_order=False, softmax_dim=-1, dropout_p=0.0,
                device='cpu', dtype=torch.float32):
    super(HiddenLayer, self).__init__()

    self.to(device=device, dtype=dtype)

    if out_features is None or out_features == 0:
        out_features = in_features
        f1 = torch.nn.Identity()
    else:
        if isinstance(in_features, list):  # bilinear (must be len = 2)
            class Bilinear(torch.nn.Module):
                def __init__(self, in1_features=in_features[0], in2_features=in_features[1],
                              out_features=out_features, bias=bias, device=device, dtype=dtype):
                    super(Bilinear, self).__init__()

                    self.F = torch.nn.Bilinear(in1_features, in2_features, out_features, bias)

                def forward(self, input):
                    input1, input2 = input
                    return self.F(input1, input2)

            f1 = Bilinear()
        else:
            f1 = torch.nn.Linear(in_features=in_features, out_features=out_features,
                                  bias=bias, device=device, dtype=dtype)

    if activation == 'identity':
        f2 = torch.nn.Identity()
    elif activation == 'polynomial':
        f2 = Polynomial(in_features=out_features, degree=degree, coef_init=coef_init,
                        coef_train=coef_train, coef_reg=coef_reg, zero_order=zero_order,
                        device=device, dtype=dtype)
    elif activation == 'tanh':
        f2 = torch.nn.Tanh()
    elif activation == 'sigmoid':
        f2 = torch.nn.Sigmoid()
    elif activation == 'softmax':
        f2 = torch.nn.Softmax(dim=softmax_dim)
    elif activation == 'relu':
        f2 = torch.nn.ReLU()
    else:
        raise ValueError(f"activation ({activation}) must be 'identity', 'polynomial', 'tanh', 'sigmoid', or 'relu'.")

    F = torch.nn.Sequential(f1, f2)

    self.dropout = torch.nn.Dropout(dropout_p)

    self.F = F
    self.device, self.dtype = device, dtype
    self.weight_reg, self.weight_norm = weight_reg, weight_norm

  def forward(self, input):
    '''
    Perform a forward pass through the hidden layer.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.

    '''
    y = self.dropout(self.F(input))
    return y

  def constrain(self):
    '''
    Constrain the weights of the hidden layer.

    '''
    for name, param in self.named_parameters():
        if 'weight' in name:
            param = torch.nn.functional.normalize(param, p=self.weight_norm, dim=1).contiguous()

  def penalize(self):
    '''
    Compute the regularization loss for the hidden layer.

    Returns:
        torch.Tensor: Regularization loss.

    '''
    loss = 0
    for name, param in self.named_parameters():
        if 'weight' in name:
            loss += self.weight_reg[0] * torch.norm(param, p=self.weight_reg[1]) * int(param.requires_grad)
        elif 'coef' in name:
            loss += self.coef_reg[0] * torch.norm(param, p=self.coef_reg[1]) * int(param.requires_grad)

    return loss
