import torch

class Criterion():
  '''
  A class for computing criterion functions.
  '''

  def __init__(self, name='mse', dims=None):
    '''
    Initializes the Criterion instance.

    Args:
        name (str): The name of the criterion function. Options are 'mae', 'mse', 'mase', 'rmse', 'nmse', 'mape', 'fb'.
        dims (int): The dimension along which the criterion is computed.
    '''
    self.name = name
    self.dims = dims

  def __call__(self, y_pred, y_true, num_params = None):
    '''
    Computes the criterion based on the predicted and true values.

    Args:
        y_pred (torch.Tensor): The predicted values.
        y_true (torch.Tensor): The true values.

    Returns:
        torch.Tensor: The computed criterion value.
    '''
    
    if self.name == 'mae':
        # Mean Absolute Error (L1 loss)
        if self.dims is not None: criterion = (y_true - y_pred).abs().nanmean(dim = self.dims)
        else: criterion = (y_true - y_pred).abs()
    elif self.name == 'mse':
        # Mean Squared Error
        if self.dims is not None: criterion = (y_true - y_pred).pow(2).nanmean(dim = self.dims)
        else: criterion = (y_true - y_pred).pow(2)
    elif self.name == 'mase':
        # Mean Absolute Scaled Error
        if self.dims is not None: criterion = (y_true - y_pred).abs().nanmean(dim=self.dims) / (y_true.diff(n=1, dim=self.dims).abs().nanmean(dim=self.dims))
        else: criterion = (y_true - y_pred).abs() / y_true.diff(n=1, dim=self.dims).abs()
    elif self.name == 'rmse':
        # Root Mean Squared Error
        if self.dims is not None: criterion = (y_true - y_pred).pow(2).nanmean(dim = self.dims).sqrt()
        else: criterion = (y_true - y_pred).pow(2).sqrt()
    elif self.name == 'nmse':
        # Normalized Mean Squared Error
        if self.dims is not None: criterion = (y_true - y_pred).pow(2).nansum(dim = self.dims) / y_true.pow(2).nansum(dim=self.dims)
        else: criterion = (y_true - y_pred).abs() / y_true.pow(2)
    elif self.name == 'mape':
        # Mean Absolute Percentage Error
        if self.dims is not None: criterion = (((y_true - y_pred) / y_true).abs() * 100).nanmean(dim = self.dims)
        else: criterion = (((y_true - y_pred) / y_true).abs() * 100)
    elif self.name == 'fb':
        # Fractional Bias
        if self.dims is not None: criterion = (y_pred.nansum(dim=self.dims) - y_true.nansum(dim=self.dims)) / y_true.nansum(dim=self.dims) * 100
        else: criterion = 2*(y_pred - y_true)/(y_pred + y_true) # (y_pred - y_true) / y_true * 100
    elif self.name == 'bic':
      N = torch.tensor(y_true.shape[0] if y_true.ndim == 2 else y_true.shape[1]).to(y_true)
      
      error_var = torch.var(y_true - y_pred)      
      criterion = N*torch.log(error_var) + num_params*torch.log(N)
    elif self.name == 'r2':
        if self.dims is not None: criterion = 1 - (y_true - y_pred).pow(2).sum(dim = self.dims)/(y_true - y_true.mean(dim = self.dims)).pow(2).sum(dim = self.dims)
        else: criterion = 1 - (y_true - y_pred).pow(2)/(y_true - y_true.mean(dim = self.dims)).pow(2)
        
    return criterion
