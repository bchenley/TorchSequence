import torch

class FeatureTransform():
  '''
  A class for performing feature scaling and transformation operations on data.
  '''

  def __init__(self,
                transform_type = 'minmax', minmax = [0., 1.], dim = 0,
                device = 'cpu', dtype = torch.float32):
    '''
    Initializes the FeatureTransform instance.

    Args:
        transform_type (str): The type of transformation to be applied. Options are 'identity', 'minmax', or 'standard'.
        minmax (list): The minimum and maximum values to transform the data when using 'minmax' transformation.
        dim (int): The dimension along which the transformation is applied.
        device (str): The device to be used for computations.
        dtype (torch.dtype): The data type to be used for computations.
    '''

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])
        
    if self.transform_type not in ['identity', 'minmax', 'standard']:
        raise ValueError(f"transform_type ({self.transform_type}) is not set to 'identity', 'minmax', or 'standard'.")

  def identity(self, X):
    '''
    Returns the input data as it is without any scaling.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The input data unchanged.
    '''
    self.min_, self.max_ = X.min(self.dim).values, X.max(self.dim).values
    return X

  def standardize(self, X):
    '''
    Performs standardization on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The standardized input data.
    '''
    self.mean_, self.std_ = X.mean(self.dim), X.std(self.dim)
    return (X - self.mean_) / self.std_

  def inverse_standardize(self, X):
    '''
    Applies inverse standardization on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The inversely standardized input data.
    '''
    return X * self.std_ + self.mean_

  def normalize(self, X):
    '''
    Performs normalization on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The normalized input data.
    '''
    self.min_, self.max_ = X.min(self.dim).values, X.max(self.dim).values
    return (X - self.min_) / (self.max_ - self.min_) * (self.minmax[1] - self.minmax[0]) + self.minmax[0]

  def inverse_normalize(self, X):
    '''
    Applies inverse normalization on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The inversely normalized input data.
    '''
    return (X - self.minmax[0]) * (self.max_ - self.min_) / (self.minmax[1] - self.minmax[0]) + self.min_

  def fit_transform(self, X):
    '''
    Fits the scaling parameters based on the input data and transforms the data accordingly.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The transformed input data.
    '''
    if self.transform_type == 'identity':
        X_transformed = self.identity(X)
    elif self.transform_type == 'minmax':
        X_transformed = self.normalize(X)
    elif self.transform_type == 'standard':
        X_transformed = self.standardize(X)

    return X_transformed

  def transform(self, X):
    '''
    Transforms the input data based on the previously fitted scaling parameters.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The transformed input data.
    '''
    if self.transform_type == 'identity':
        X_transformed = X
    elif self.transform_type == 'minmax':
        X_transformed = (X - self.min_) / (self.max_ - self.min_) * (self.minmax[1] - self.minmax[0]) + self.minmax[0]
    elif self.transform_type == 'standard':
        X_transformed = (X - self.mean_) / self.std_

    return X_transformed

  def inverse_transform(self, X):
    '''
    Applies the inverse transformation on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The inversely transformed input data.
    '''
    if self.transform_type == 'identity':
        X_inverse_transformed = X
    elif self.transform_type == 'minmax':
        X_inverse_transformed = self.inverse_normalize(X)
    elif self.transform_type == 'standard':
        X_inverse_transformed = self.inverse_standardize(X)

    return X_inverse_transformed
