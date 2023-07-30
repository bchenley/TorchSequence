import pandas as pd
import numpy as np

import yfinance as yf

def load_yfinance(symbols,
                  start_time,
                  end_time = None,
                  interval = '1d',
                  date_format = "%y-%m-%d"):

  end_time = end_time or datetime.now().strftime(date_format)

  df = yf.download(tickers = symbols,
                   start = start_time,
                   end = end_time,
                   group_by = 'ticker',
                   interval = interval,
                   progress = False).reset_index()

  date_df = df.filter(regex = 'Date')
  date_df.columns = ['date']
  date_df['date'] = pd.to_datetime(date_df['date'])

  df = df.drop(columns=df.filter(regex='Date').columns)

  if len(symbols) > 1:
    df.columns = df.columns.get_level_values(0) + '_' + df.columns.get_level_values(1).str.lower().str.replace(' ', '_')
  else:
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df.columns = ['_'.join([symbols[0],attr]) for attr in list(df.columns)]

  df = pd.concat([date_df, df], axis = 1)

  return df
