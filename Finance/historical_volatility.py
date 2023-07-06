import numpy as np
import pandas as pd

def historical_volatility(df, months):

  window_size = 21*months

  hv = []
  for window in df.rolling(window=window_size):
    if window.shape[0] < window_size:
      x = np.nan
    else:
      ssum = window.diff().dropna().apply(lambda x: x*x).sum()
      x = np.sqrt(252 * ssum / window_size) # * 100
    hv.append(x)

  return hv
