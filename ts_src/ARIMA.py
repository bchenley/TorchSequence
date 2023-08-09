import statsmodels.api as sm
import torch
import numpy as np
import pandas as pd

from ts_src.Criterion import Criterion 

class ARIMA():
  def __init__(self,
               df_train, df_test,
               endog_name, exog_name = None, 
               order = (0, 0, 0),
               loss = 'mse', metric = 'mae'):

    self.df_train, self.df_test = df_train.copy(), df_test.copy()
    self.endog_name, self.exog_name = endog_name, exog_name
    self.order = order

    self.loss_fn = Criterion(loss) if loss is not None else None
    self.metric_fn = Criterion(metric) if metric is not None else None

  def fit(self):

    self.model = sm.tsa.ARIMA(endog = self.df_train[self.endog_name], 
                              exog = self.df_train[self.exog_name] if self.exog_name is not None else None,                              
                              order = self.order)
        
    self.results = self.model.fit()
  
  def predict(self, transforms = None):

    if not hasattr(self, 'results'): self.fit()

    p, d, q = self.order

    df_test = self.df_test.copy()
    
    # test
    df_test[f"{self.endog_name}_prediction"] = np.full((df_test.shape[0],), np.nan) 
    
    test_target = df_test[self.endog_name].values
    
    X0 = []
    for _ in range(d):
      X0.insert(0, test_target[:1])
      test_target = np.pad(np.diff(test_target, 1, 0), (1, 0))

    test_target = np.pad(test_target, (p, 0), mode = 'constant') 

    for n in range(p, test_target.shape[0]):
      df_test[f"{self.endog_name}_prediction"][n-p] = np.dot(np.flip(test_target[(n-p):n], axis=0), self.results.arparams)
    #
    
    for i in range(d):
      df_test[f"{self.endog_name}_prediction"] = np.cumsum(df_test[f"{self.endog_name}_prediction"],0) + X0[i]

    if transforms is not None:
      if self.endog_name in transforms:
        df_test[self.endog_name] = transforms[self.endog_name].inverse_transform(torch.tensor(df_test[self.endog_name].values)).cpu().numpy()
        df_test[f"{self.endog_name}_prediction"] = transforms[self.endog_name].inverse_transform(torch.tensor(df_test[f"{self.endog_name}_prediction"].values)).cpu().numpy()
    
    if self.loss_fn is not None:
      df_test[f"{self.endog_name}_{self.loss_fn.name}"] = self.loss_fn(torch.tensor(df_test[f"{self.endog_name}_prediction"].values),
                                                                       torch.tensor(df_test[self.endog_name].values)).cpu().numpy()
      
    if self.metric_fn is not None:
      df_test[f"{self.endog_name}_{self.metric_fn.name}"] = self.metric_fn(torch.tensor(df_test[f"{self.endog_name}_prediction"].values),
                                                                            torch.tensor(df_test[self.endog_name].values)).cpu().numpy()
      
    return df_test

  def forecast(self, 
               num_forecast_steps = 1,
               input = None,
               transforms = None):

    if not hasattr(self, 'results'): self.fit()
    
    p, d, q = self.order
    
    if input is not None:  
      input_ = input.clone()[:, -p:] if isinstance(input, torch.Tensor) else input.copy()[:, -p:]
    else:
      input_ = self.df_test[self.endog_name].copy()[-p:].values.reshape(1, p, 1)

    if not isinstance(input_, torch.Tensor): input_ = torch.tensor(input_)

    X0 = []
    for _ in range(d):
      X0.insert(0, input_[:, :1])      
      input_ = torch.nn.functional.pad(input_.diff(1, 1), (0, 0, 1, 0, 0, 0), value = 0)

    num_samples = input_.shape[0]
    
    arparams = torch.tensor(self.results.arparams.reshape(1, p, 1).repeat(num_samples, axis = 0)).to(input_)

    forecast = []
    for n in range(num_forecast_steps):
      
      forecast_n = torch.matmul(input_.flip(1).permute(0, 2, 1), arparams).permute(0, 2, 1)

      forecast.append(forecast_n)
      
      input_ = torch.cat((input_[:, 1:], forecast_n), 1)

    forecast = torch.cat(forecast, 1)

    for i in range(d):
      forecast = forecast.cumsum(1) + X0[i]
    
    if transforms is not None:
      if self.endog_name in transforms:
        for i, batch in enumerate(forecast):
          forecast[i] = transforms[self.endog_name].inverse_transform(batch)

    return forecast
