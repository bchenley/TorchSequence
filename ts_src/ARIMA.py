import statsmodels as sm
import numpy as np

class ARIMA():
  def __init__(self,
               df_train, df_test, 
               endog_name, 
               exog_name = None, order = (0, 0, 0)):

    locals_ = locals().copy()
    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])

    self.model = sm.tsa.arima.model.ARIMA(endog = df_train[endog_name], 
                                          exog = df_train[exog_name] if exog_name is not None else None,                              
                                          order = self.order)
  
  def fit(self):

    self.results = model.fit()
  
  def predict(self):

    if not hasattr(self, 'results'): self.fit()

    p, d, q = self.order

    # train
    self.df_train[f"{self.endog_name}_prediction"] = np.full((self.df_train.shape[0],), np.nan) 

    train_target = np.pad(self.df_train[self.endog_name].values, (p, 0), mode = 'constant') 

    for n in range(p, train_target.shape[0]):
      self.df_train[f"{self.endog_name}_prediction"][n-p] = np.dot(np.flip(train_target[(n-p):n], axis=0), self.results.arparams)
    #

    # test
    self.df_test[f"{self.endog_name}_prediction"] = np.full((self.df_test.shape[0],), np.nan) 

    test_target = np.pad(self.df_test[self.endog_name].values, (p, 0), mode = 'constant') 

    for n in range(p, test_target.shape[0]):
      self.df_test[f"{self.endog_name}_prediction"][n-p] = np.dot(np.flip(test_target[(n-p):n], axis=0), self.results.arparams)
    #
    
  def forecast(self, num_forecast_steps = 1, input = None):

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

    return forecast
