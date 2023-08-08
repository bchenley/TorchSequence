import statsmodels as sm 
import numpy as np

from ts_src.Criterion import Criterion 

class ARIMA():
  def __init__(self,
               df_train, df_test,
               endog_name, 
               exog_name = None, order = (0, 0, 0),
               loss = 'mse', metric = 'mae'):

    locals_ = locals().copy()
    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])

    self.model = sm.tsa.arima.model.ARIMA(endog = df_train[endog_name], 
                                          exog = df_train[exog_name] if exog_name is not None else None,                              
                                          order = self.order)

    self.loss_fn = Criterion(self.loss) if self.loss is not None else None
    self.metric_fn = Criterion(self.metric) if self.metric is not None else None

  def fit(self, transforms = None):
    
    if transforms is not None:
      for name in [self.endog_name, self.exog_name]:
        if name in transforms:
          self.df_train[name] = transforms[name].fit_transform(self.df_train[name])
          self.df_test[name] = transforms[name].transform(self.df_test[name])

    self.results = model.fit()
  
  def predict(self, transforms = None):

    if not hasattr(self, 'results'): self.fit()

    p, d, q = self.order

    df_train, df_test = self.df_train.copy(), self.df_test.copy()

    # train
    df_train[f"{self.endog_name}_prediction"] = np.full((df_train.shape[0],), np.nan) 

    train_target = np.pad(df_train[self.endog_name].values, (p, 0), mode = 'constant') 

    for n in range(p, train_target.shape[0]):
      df_train[f"{self.endog_name}_prediction"][n-p] = np.dot(np.flip(train_target[(n-p):n], axis=0), self.results.arparams)
    #

    # test
    df_test[f"{self.endog_name}_prediction"] = np.full((df_test.shape[0],), np.nan) 

    test_target = np.pad(df_test[self.endog_name].values, (p, 0), mode = 'constant') 

    for n in range(p, test_target.shape[0]):
      df_test[f"{self.endog_name}_prediction"][n-p] = np.dot(np.flip(test_target[(n-p):n], axis=0), self.results.arparams)
    #

    if transforms is not None:
      if self.endog_name in transforms:
        df_train[self.endog_name] = transforms[self.endog_name].inverse_transform(df_train[self.endog_name])
        df_train[f"{self.endog_name}_prediction"] = transforms[self.endog_name].inverse_transform(df_train[f"{self.endog_name}_prediction"])

        df_test[self.endog_name] = transforms[self.endog_name].inverse_transform(df_test[self.endog_name])
        df_test[f"{self.endog_name}_prediction"] = transforms[self.endog_name].inverse_transform(df_test[f"{self.endog_name}_prediction"])

    if self.loss_fn is not None:
      df_train[f"{self.endog_name}_{self.loss_fn.name}"] = self.loss_fn(df_train[f"{self.endog_name}_prediction"],
                                                                        df_train[self.endog_name])
      df_test[f"{self.endog_name}_{self.loss_fn.name}"] = self.loss_fn(df_test[f"{self.endog_name}_prediction"],
                                                                       df_test[self.endog_name])
      
    if self.metric_fn is not None:
      df_train[f"{self.endog_name}_{self.metric_fn.name}"] = self.metric_fn(df_train[f"{self.endog_name}_prediction"],
                                                                            df_train[self.endog_name])
      df_test[f"{self.endog_name}_{self.metric_fn.name}"] = self.metric_fn(df_test[f"{self.endog_name}_prediction"],
                                                                           df_test[self.endog_name])
    
    return df_train, df_test

  def forecast(self, 
               num_forecast_steps = 1,
               input = None,
               transforms = None):

    if not hasattr(self, 'results'): self.fit()
    
    p, d, q = self.order

    if input is not None:  
      input_ = input.copy()[:, -p:, :1]
    else:
      input_ = self.df_test[self.endog_name][-p:].values.reshape(1, p, 1)

    num_samples = input_.shape[0]

    arparams = self.results.arparams.reshape(1, p, 1).repeat(num_samples, axis = 0)
    
    forecast = []
    for n in range(num_forecast_steps):
      
      forecast_n = np.matmul(np.flip(input_.transpose(0, 2, 1), axis = 2), arparams).transpose(0, 2, 1)

      forecast.append(forecast_n)

      input_ = np.concatenate((input_[:, 1:], forecast_n), 1)

    forecast = np.concatenate(forecast, 1)

    if transforms is not None:
      if self.endog_name in transforms:
        forecast = transforms[self.endog_name].inverse_transform(forecast)
        
    return forecast
