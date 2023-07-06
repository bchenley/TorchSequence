import pandas as pd
import numpy as np
  
def daily_volatility(df, days, interval = '1h'):

  if interval == '1h':
    k = 6
  elif interval == '1m':
    k = 360
  else:
    raise ValueError(f"interval ({interval}) can only be '1h' or '1m'")

  window_size = k*days

  dv = df.diff().rolling(window=window_size).std()

  # dv = []
  # for window in df.diff().rolling(window=window_size):
  #   if window.shape[0] < window_size:
  #       x = np.nan
  #   else:
  #     dv.append(window.std())

  return dv
