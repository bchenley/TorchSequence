class Criterion():
  '''
  A class for computing criterion functions.
  '''

  def __init__(self, name='mse', dims=0):
    '''
    Initializes the Criterion instance.

    Args:
        name (str): The name of the criterion function. Options are 'mae', 'mse', 'mase', 'rmse', 'nmse', 'mape', 'fb'.
        dims (int): The dimension along which the criterion is computed.
    '''
    self.name = name
    self.dims = dims

  def __call__(self, y_pred, y_true):
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
        criterion = (y_true - y_pred).abs().nanmean(dim=self.dims)
    elif self.name == 'mse':
        # Mean Squared Error
        criterion = (y_true - y_pred).pow(2).nanmean(dim=self.dims)
    elif self.name == 'mase':
        # Mean Absolute Scaled Error
        criterion = (y_true - y_pred).abs().nanmean(dim=self.dims) / (y_true.diff(n=1, dim=self.dims).abs().nanmean(dim=self.dims))
    elif self.name == 'rmse':
        # Root Mean Squared Error
        criterion = (y_true - y_pred).pow(2).nanmean(dim=self.dims).sqrt()
    elif self.name == 'nmse':
        # Normalized Mean Squared Error
        criterion = (y_true - y_pred).pow(2).nanmean(dim=self.dims) / y_true.pow(2).nanmean(dim=self.dims)
    elif self.name == 'mape':
        # Mean Absolute Percentage Error
        criterion = (((y_true - y_pred) / y_true).abs() * 100).nanmean(dim=self.dims)
    elif self.name == 'fb':
        # Fractional Bias
        criterion = (y_pred.nansum(dim=self.dims) - y_true.nansum(dim=self.dims)) / y_true.nansum(dim=self.dims)

    return criterion
