
import torch
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
    
    df = self.df.copy()

    df[f"{self.endog_name}_prediction"] = torch.full((self.df.shape[0], ), torch.nan)
    
    for n in range(self.naive_steps, df.shape[0]):
      df.loc[n, f"{self.endog_name}_prediction"] = df.loc[n - self.naive_steps, self.endog_name]

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

    if input is not None:
      input_ = input.clone()[:, -self.naive_steps:] if isinstance(input, torch.Tensor) else input.copy()[:, -self.naive_steps:]
    else:
      input_ = self.df[self.endog_name].copy()[-self.naive_steps:].values.reshape(1, self.naive_steps, 1)

    if not isinstance(input_, torch.Tensor): input_ = torch.tensor(input_)

    num_samples = input_.shape[0]

    forecast = []
    for n in range(self.naive_steps, num_forecast_steps + self.naive_steps):
      
      forecast_n = input_[:, (n-self.naive_steps):(n-self.naive_steps+1)]
      
      forecast.append(forecast_n)

      input_ = torch.cat((input_[:, 1:], forecast_n), 1)

    forecast = torch.cat(forecast, 1)

    if transforms is not None:
      if self.endog_name in transforms:
        for i, batch in enumerate(forecast):
          forecast[i] = transforms[self.endog_name].inverse_transform(batch)

    return forecast
