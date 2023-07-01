class Loss():
  '''
  A class for computing loss functions.
  '''

  def __init__(self, name='mse', dims=0):
    '''
    Initializes the Loss instance.

    Args:
        name (str): The name of the loss function. Options are 'mae', 'mse', 'mase', 'rmse', 'nmse', 'mape', 'fb'.
        dims (int): The dimension along which the loss is computed.
    '''
    self.name = name
    self.dims = dims

  def __call__(self, y_pred, y_true):
    '''
    Computes the loss based on the predicted and true values.

    Args:
        y_pred (torch.Tensor): The predicted values.
        y_true (torch.Tensor): The true values.

    Returns:
        torch.Tensor: The computed loss value.
    '''
    if self.name == 'mae':
        # Mean Absolute Error (L1 loss)
        loss = (y_true - y_pred).abs().nanmean(dim=self.dims)
    elif self.name == 'mse':
        # Mean Squared Error
        loss = (y_true - y_pred).pow(2).nanmean(dim=self.dims)
    elif self.name == 'mase':
        # Mean Absolute Scaled Error
        loss = (y_true - y_pred).abs().nanmean(dim=self.dims) / (y_true.diff(n=1, dim=self.dims).abs().nanmean(dim=self.dims))
    elif self.name == 'rmse':
        # Root Mean Squared Error
        loss = (y_true - y_pred).pow(2).nanmean(dim=self.dims).sqrt()
    elif self.name == 'nmse':
        # Normalized Mean Squared Error
        loss = (y_true - y_pred).pow(2).nanmean(dim=self.dims) / y_true.pow(2).nanmean(dim=self.dims)
    elif self.name == 'mape':
        # Mean Absolute Percentage Error
        loss = (((y_true - y_pred) / y_true).abs() * 100).nanmean(dim=self.dims)
    elif self.name == 'fb':
        # Fractional Bias
        loss = (y_pred.nansum(dim=self.dims) - y_true.nansum(dim=self.dims)) / y_true.nansum(dim=self.dims)

    return loss
