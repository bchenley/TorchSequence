
import numpy as np
import pandas as pd

from ts_src.Criterion import Criterion 

class Naive():
  def __init__(self,
               df,
               endog_name,
               naive_steps = 1,
               loss = 'mse', metric = 'mae'):
    
    self.df = df.copy().reset_index(drop = True)
    self.endog_name = endog_name
    self.naive_steps = naive_steps

    self.loss_fn = Criterion(loss) if loss is not None else None
    self.metric_fn = Criterion(metric) if metric is not None else None
  
  def predict(self, transforms = None):
    
    self.df[f"{self.endog_name}_prediction"] = np.full((self.df.shape[0], ), np.nan)
    
    for n in range(self.naive_steps, self.df.shape[0]):
      self.df.loc[n, f"{self.endog_name}_prediction"] = self.df.loc[n - self.naive_steps, self.endog_name]

    if transforms is not None:
      if self.endog_name in transforms:
        self.df[self.endog_name] = transforms[self.endog_name].inverse_transform(self.df[self.endog_name].values).cpu().numpy()
        self.df[f"{self.endog_name}_prediction"] = transforms[self.endog_name].inverse_transform(self.df[f"{self.endog_name}_prediction"].values).cpu().numpy()

    if self.loss_fn is not None:
      self.df[f"{self.endog_name}_{self.loss_fn.name}"] = self.loss_fn(self.df[f"{self.endog_name}_prediction"].values,
                                                                       self.df[self.endog_name].values)
      
    if self.metric_fn is not None:
      self.df[f"{self.endog_name}_{self.metric_fn.name}"] = self.metric_fn(self.df[f"{self.endog_name}_prediction"].values,
                                                                           self.df[self.endog_name].values)
       
  def forecast(self, 
               num_forecast_steps = 1, 
               input = None,
               transforms = None):

    if input is not None:
      input_ = input.copy()[:, -num_forecast_steps:, :1]
    else:
      input_ = self.df[self.endog_name][-num_forecast_steps:].values.reshape(1, num_forecast_steps, 1)

    num_samples = input_.shape[0]

    forecast = []
    for n in range(self.naive_steps, num_forecast_steps + self.naive_steps):

      forecast_n = input_[:, (n-self.naive_steps):(n-self.naive_steps+1)]
      
      forecast.append(forecast_n)

      input_ = np.concatenate((input_[:, 1:], forecast_n), 1)

    forecast = np.concatenate(forecast, 1)

    if transforms is not None:
      if self.endog_name in transforms:
        forecast = transforms[self.endog_name].inverse_transform(forecast).cpu().numpy()

    return forecast
