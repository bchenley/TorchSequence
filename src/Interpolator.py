import scipy as sc

class Interpolator():
  '''
  Interpolator for 1-dimensional data.

  Args:
      kind: The kind of interpolation method to use.
      axis: The axis along which to interpolate.

  Attributes:
      interp_fn: The interpolation function.

  Methods:
      fit: Fits the interpolation function to the provided data.

  '''
  def __init__(self, kind='linear', axis=0):
      super().__init__()

      self.kind = kind
      self.axis = axis
      self.interp_fn = None

  def fit(self, x, y):
      '''
      Fits the interpolation function to the provided data.

      Args:
          x: The x-coordinates of the data points.
          y: The y-coordinates of the data points.

      '''
      if isinstance(x, torch.Tensor):
          x = x.detach().numpy()
      if isinstance(y, torch.Tensor):
          y = y.detach().numpy()

      self.interp_fn = sc.interpolate.interp1d(x, y, kind=self.kind, axis=self.axis)
