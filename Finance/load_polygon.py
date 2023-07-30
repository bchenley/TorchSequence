import pandas as pd
import numpy as np
import requests
import json

def load_polygon(apiKey,
                 symbols,
                 start_time,
                 end_time = None,
                 # datetime_unit = 'D',
                 date_format = "%y-%m-%d",
                 parsing = 'day'):

  end_time = end_time or datetime.now().strftime(date_format)

  df = None
  for symbol in symbols:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{parsing}/{str(start_time)}/{str(end_time)}?apiKey={apiKey}"

    page = requests.get(url)

    if page.status_code == 200:
      content = json.loads(page.content)
      total = content["resultsCount"]
      data = content["results"]

      if df is None:
        df = pd.DataFrame(data)
        df.rename(columns={"o": f"{symbol}_adj_open",
                          "c": f"{symbol}_adj_close",
                          "l": f"{symbol}_adj_low",
                          "h": f"{symbol}_adj_high",
                          "v": f"{symbol}_adj_volume",
                          "n": f"{symbol}_number",
                          "vw": f"{symbol}_vwap",
                          "t": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]/1_000, unit="s")

        df.sort_values("date", inplace=True)
        df.drop_duplicates(subset="date", inplace=True)
        df.insert(0, 'date', df.pop('date'))
      else:
        df_ = pd.DataFrame(data)
        df_.rename(columns={"o": f"{symbol}_adj_open",
                            "c": f"{symbol}_adj_close",
                            "l": f"{symbol}_adj_low",
                            "h": f"{symbol}_adj_high",
                            "v": f"{symbol}_adj_volume",
                            "n": f"{symbol}_number",
                            "vw": f"{symbol}_vwap",
                            "t": "date"}, inplace=True)
        df_["date"] = pd.to_datetime(df_["date"]/1_000, unit="s") 

        df_.sort_values("date", inplace=True)
        df_.drop_duplicates(subset="date", inplace=True)

        df = pd.merge(df, df_, on = 'date', how = 'inner')
    else:
      raise ValueError(f"Failed to load {url} (error code {page.status_code})\nPage Content: {page.content}")

    df['date'] = pd.to_datetime(df['date'])

  return df
