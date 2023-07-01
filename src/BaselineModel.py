import numpy as np

class BaselineModel():
  '''
  Baseline models for time series prediction.

  Args:
      model_type (str): Type of baseline model.
      naive_steps (int): Number of steps for the naive baseline model.
      ma_window_size (int): Moving average window size for the moving average baseline model.
      decay (float): Decay factor for exponential smoothing models.
      trend (list): Trend parameters for exponential smoothing models.
      period (int): Period parameter for seasonal exponential smoothing model.
      seasonal (list): Seasonal parameters for seasonal exponential smoothing model.

  '''

  def __init__(self, model_type='naive', naive_steps=1, ma_window_size=20, decay=0.5, trend=[0.5, 1.0], period=1, seasonal=[0.5, 1.0]):
      self.model_type = model_type
      self.naive_steps = naive_steps
      self.decay, self.trend, self.seasonal, self.period = decay, trend, seasonal, period
      self.ma_window_size = ma_window_size

  def ma_prediction(self, input):
      '''
      Moving average prediction.

      Args:
          input: The input data tensor.

      Returns:
          prediction: The predicted values based on the moving average model.

      '''
      prediction = []
      for n in range(input.shape[0]):
          prediction_n = input[np.max([0, n - self.ma_window_size]):n].mean(0, keepdims=True)
          prediction.append(prediction_n)

      prediction = torch.cat(prediction, 0)

      return prediction

  def naive_prediction(self, input):
    '''
    Naive prediction.

    Args:
        input: The input data tensor.

    Returns:
        prediction: The predicted values based on the naive model.

    '''
    prediction = torch.full((self.naive_steps, input.shape[1]), float('nan')).to(input)
    for n in range(self.naive_steps, input.shape[0]):
        prediction_n = input[n - self.naive_steps]
        prediction = torch.cat((prediction, prediction_n), 0)

    return prediction

  def ses_prediction(self, input):
    '''
    Single exponential smoothing prediction.

    Args:
        input: The input data tensor.

    Returns:
        prediction: The predicted values based on the single exponential smoothing model.

    '''
    prediction = torch.full((1, input.shape[1]), float('nan')).to(input)
    for n in range(1, input.shape[0]):
        prediction_n = self.decay * input[n - 1] + (1 - self.decay) * prediction_n[n - 1]
        prediction = torch.cat((prediction, prediction_n), 0)

  def des_prediction(self, input):
    '''
    Double exponential smoothing prediction.

    Args:
        input: The input data tensor.

    Returns:
        prediction: The predicted values based on the double exponential smoothing model.

    '''
    prediction = torch.full((1, input.shape[1]), float('nan')).to(input)
    level_prev, trend_prev = 0, 0
    for n in range(1, input.shape[0]):
        level_n = self.decay * input[n - 1] + (1 - self.decay) * (level_prev + trend_prev)
        trend_n = self.trend[0] * (level_n - level_prev) + (1 - self.trend[0]) * trend_prev

        prediction_n = level_n + self.trend[1] * trend_n

        prediction = torch.cat((prediction, prediction_n), 0)

        level_prev, trend_prev = level_n, trend_prev

    return prediction

  def tes_prediction(self, input):
    '''
    Seasonal exponential smoothing prediction.

    Args:
        input: The input data tensor.

    Returns:
        prediction: The predicted values based on the seasonal exponential smoothing model.

    '''
    prediction = torch.full((self.period, input.shape[1]), float('nan')).to(input)
    level_prev, trend_prev = 0, 0
    season = torch.zeros_like(input).to(input)
    for n in range(self.period, input.shape[0]):
        level_n = self.decay * input[n - 1] + (1 - self.decay) * (level_prev + trend_prev) + season[n - self.period]
        trend_n = self.trend[0] * (level_n - level_prev) + (1 - self.trend[0]) * trend_prev
        season[n] = self.seasonal[0] * (input[n - 1] - level_prev - trend_prev) + (1 - self.seasonal) * season[
            n - self.period]

        prediction_n = level_n + self.trend[1] * trend_n + self.seasonal[1] * season[n]

        prediction = torch.cat((prediction, prediction_n), 0)

        level_prev, trend_prev = level_n, trend_prev

  def __call__(self, input):
    '''
    Make predictions based on the selected model type.

    Args:
        input: The input data tensor.

    Returns:
        prediction: The predicted values based on the selected model type.

    '''
    if self.model_type == 'naive':
        return self.naive_prediction(input)

    if self.model_type == 'moving_average':
        return self.ma_prediction(input)

    if self.model_type == 'bayesian':
        return self.bayesian_prediction(input)

    if self.model_type == 'ses':
        return self.ses_prediction(input)

    if self.model_type == 'des':
        return self.des_prediction(input)
