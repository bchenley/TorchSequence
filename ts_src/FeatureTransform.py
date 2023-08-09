import torch

class FeatureTransform():
  '''
  A class for performing feature scaling and transformation operations on data.
  '''

  def __init__(self,
               transform_type = 'minmax', minmax = [0., 1.], dim = 0,
               diff_order = 0,  
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

    if self.transform_type == 'identity':
        self.transform_fn = self.identity
        self.inverse_transform_fn = self.inverse_identity
    elif self.transform_type == 'minmax':
        self.transform_fn = self.normalize
        self.inverse_transform_fn = self.inverse_normalize
    elif self.transform_type == 'standard':
        self.transform_fn = self.standardize
        self.inverse_transform_fn = self.inverse_standardize
      
  def identity(self, X, fit = False):
    '''
    Returns the input data as it is without any scaling.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The input data unchanged.
    '''
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X
    
    X = self.difference(X) if self.diff_order > 0 else X
    if fit: self.min_, self.max_ = X.min(self.dim).values, X.max(self.dim).values
    return X
  
  def inverse_identity(self, X):
    
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X
    
    return self.cumsum(X) if self.diff_order > 0 else X

  def difference(self, X, fit = False):
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    y = X.clone()
    self.X0 = []
    for i in range(self.diff_order):
      self.X0.append(y[:1])
      y = y.diff(1, self.dim)
    y = torch.nn.functional.pad(y, (0, 0, self.diff_order, 0), mode = 'constant', value = 0)
  
    return y

  def cumsum(self, X):
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    y = X.clone()
    y = y[self.diff_order:]
    for i in range(self.diff_order):
      y = torch.cat((self.X0[-(i+1)], y), self.dim).cumsum(self.dim)

    return y
  
  def standardize(self, X, fit = False):
    '''
    Performs standardization on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The standardized input data.
    '''
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    X = self.difference(X) if self.diff_order > 0 else X
    
    if fit: self.mean_, self.std_ = X.mean(self.dim), X.std(self.dim)
    
    return (X - self.mean_) / self.std_

  def inverse_standardize(self, X):
    '''
    Applies inverse standardization on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The inversely standardized input data.
    '''
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    y = X * self.std_ + self.mean_
    
    y = self.cumsum(y) if self.diff_order > 0 else y
    
    return y

  def normalize(self, X, fit = False):
    '''
    Performs normalization on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The normalized input data.
    '''
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    X = self.difference(X) if self.diff_order > 0 else X
    
    if fit: self.min_, self.max_ = X.min(self.dim).values, X.max(self.dim).values
      
    return (X - self.min_) / (self.max_ - self.min_) * (self.minmax[1] - self.minmax[0]) + self.minmax[0]

  def inverse_normalize(self, X):
    '''
    Applies inverse normalization on the input data.

    Args:
        X (torch.Tensor): The input data.
    
    Returns:
        torch.Tensor: The inversely normalized input data.
    '''
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    y = (X - self.minmax[0]) * (self.max_ - self.min_) / (self.minmax[1] - self.minmax[0]) + self.min_
    
    y = self.cumsum(y) if self.diff_order > 0 else y
    
    return y

  def fit_transform(self, X):
    '''
    Fits the scaling parameters based on the input data and transforms the data accordingly.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The transformed input data.
    '''    
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    return self.transform_fn(X, True)

  def transform(self, X):
    '''
    Transforms the input data based on the previously fitted scaling parameters.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The transformed input data.
    '''
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    return self.transform_fn(X)

  def inverse_transform(self, X):
    '''
    Applies the inverse transformation on the input data.

    Args:
        X (torch.Tensor): The input data.

    Returns:
        torch.Tensor: The inversely transformed input data.
    '''
    X = torch.tensor(X).to(device = self.device, dtype = self.dtype) if not isinstance(X, torch.Tensor) else X.to(device = self.device, dtype = self.dtype)
    
    return self.inverse_transform_fn(X)
