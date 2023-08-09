import torch
import pandas as pd

from ts_src.Criterion import Criterion

class MovingAverage():
  """
  A class for implementing Moving Average forecasting.

  Parameters:
  - df (pd.DataFrame): The dataframe containing the time series data.
  - endog_name (str): The name of the endogenous variable (target variable).
  - window_type (str): The type of window to use for the moving average.
  - window_len (int): The length of the moving average window.
  - loss (str): The loss function for evaluation.
  - metric (str): The evaluation metric.
  """

  def __init__(self,
               df, 
               endog_name,
               window_type = None, window_len = 1,
               loss = 'mse', metric = 'mae'):
    
    self.df = df.copy().reset_index(drop = True)
    self.endog_name = endog_name
    self.window_type = window_type
    self.window_len = window_len

    if self.window_type == 'hanning':
      self.window = torch.hanning(self.window_len)
    elif self.window_type == 'hamming':
      self.window = torch.hamming(self.window_len)
    else:
      self.window = None
    
    self.loss_fn = Criterion(loss) if loss is not None else None
    self.metric_fn = Criterion(metric) if metric is not None else None
  
  def predict(self, transforms = None):
    
    """
    Calculate and store the moving average predictions.

    Parameters:
    - transforms (dict): Dictionary of transformations to apply.

    Returns:
    - None
    """

    df = self.df.copy()

    df[f"{self.endog_name}_prediction"] = torch.full((self.df.shape[0], ), torch.nan)

    for n in range(self.window_len, df.shape[0]):      
      input_n = df[self.endog_name].values[(n - self.window_len):n]

      input_n = input_n * self.window / self.window.sum() if self.window is not None else input_n
      
      df.loc[n, f"{self.endog_name}_prediction"] = input_n.mean(0)

    if transforms is not None:
      if self.endog_name in transforms:
        df[self.endog_name] = transforms[self.endog_name].inverse_transform(df[self.endog_name].values).cpu().numpy()        
        df[f"{self.endog_name}_prediction"] = transforms[self.endog_name].inverse_transform(df[f"{self.endog_name}_prediction"].values).cpu().numpy()
        
    if self.loss_fn is not None:
      df[f"{self.endog_name}_{self.loss_fn.name}"] = self.loss_fn(torch.tensor(df[f"{self.endog_name}_prediction"].values),
                                                                  torch.tensor(df[self.endog_name].values)).numpy()
      
    if self.metric_fn is not None:
      df[f"{self.endog_name}_{self.metric_fn.name}"] = self.metric_fn(torch.tensor(df[f"{self.endog_name}_prediction"].values),
                                                                      torch.tensor(df[self.endog_name].values)).numpy()
    
    return df

  def forecast(self, 
               num_forecast_steps = 1,
               input = None,
               transforms = None):
    """
    Generate forecasts using the moving average model.

    Parameters:
    - num_forecast_steps (int): Number of forecast steps to generate.
    - input (torch.Tensor or np.ndarray): Input data for forecasting.
    - transforms (dict): Dictionary of transformations to apply.

    Returns:
    - forecast (torch.Tensor): Forecasted values.
    """
    if input is not None:
      input_ = input.clone()[:, -self.window_len:] if isinstance(input, torch.Tensor) else input.copy()[:, -self.window_len:]
    else:
      input_ = self.df[self.endog_name].copy()[-self.window_len:].values.reshape(1, self.window_len, 1)

    if not isinstance(input_, torch.Tensor): input_ = torch.tensor(input_)

    num_samples = input_.shape[0]
    
    forecast = []
    for n in range(self.window_len, num_forecast_steps + self.window_len):
      
      input_n = input_[:, (n - self.window_len):n]
      input_n = input_n * self.window / self.window.sum() if self.window is not None else input_n
      
      forecast_n = input_n.mean(1).reshape(num_samples, 1, -1)
      
      forecast.append(forecast_n)

      input_ = torch.cat((input_[:, 1:], forecast_n), 1)

    forecast = torch.cat(forecast, 1)

    if transforms is not None:
      if self.endog_name in transforms:
        for i, batch in enumerate(forecast):
          forecast[i] = transforms[self.endog_name].inverse_transform(batch)

    return forecast
