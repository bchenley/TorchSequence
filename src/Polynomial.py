import torch 

class Polynomial(torch.nn.Module):
  '''
  Polynomial regression model.

  Args:
  - in_features (int): Number of input features.
  - degree (int): Degree of the polynomial.
  - coef_init (torch.Tensor): Initial coefficients for the polynomial. If None, coefficients are initialized randomly.
  - coef_train (bool): Whether to train the coefficients.
  - coef_reg (list): Regularization parameters for the coefficients. [Regularization weight, regularization exponent]
  - zero_order (bool): Whether to include the zeroth-order term (constant) in the polynomial.
  - device (str): Device to use for computation ('cpu' or 'cuda').
  - dtype (torch.dtype): Data type of the coefficients.
  '''

  def __init__(self,
               in_features, degree=1, coef_init=None, coef_train=True,
               coef_reg=[0.001, 1], zero_order=True,
               device='cpu', dtype=torch.float32):
      super(Polynomial, self).__init__()

      self.to(device=device, dtype=dtype)

      if coef_init is None:
          coef_init = torch.nn.init.normal_(torch.empty(in_features, degree + int(zero_order)))

      coef = torch.nn.Parameter(data=coef_init.to(device=device, dtype=dtype), requires_grad=coef_train)

      self.coef, self.coef_reg = coef, coef_reg
      self.in_features, self.degree = in_features, degree
      self.zero_order = zero_order
      self.device, self.dtype = device, dtype

  def forward(self, X):
    '''
    Perform forward pass to compute polynomial regression.

    Args:
    - X (torch.Tensor): Input data tensor of shape (batch_size, in_features).

    Returns:
    - y (torch.Tensor): Output predictions of shape (batch_size).
    '''

    X = X.to(device=self.device, dtype=self.dtype)

    pows = torch.arange(1 - int(self.zero_order), (self.degree + 1), device=self.device, dtype=self.dtype)

    y = (X.unsqueeze(-1).pow(pows) * self.coef).sum(-1)

    return y

  def penalize(self):
    '''
    Compute the penalty term for coefficient regularization.

    Returns:
    - penalty (torch.Tensor): Penalty term based on coefficient regularization.
    '''

    return self.coef_reg[0] * torch.norm(self.coef, p=self.coef_reg[1]) * int(self.coef.requires_grad)
