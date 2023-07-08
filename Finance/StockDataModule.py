import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from datetime import datetime

from TorchTimeSeries.src import SequenceDataloader
from Finance.load_polygon import load_polygon
from Finance.load_yfinance import load_yfinance
from Finance.historical_volatility import historical_volatility
from Finance.daily_volatility import daily_volatility

class StockDataModule(pl.LightningDataModule):
  def __init__(self,
               source,
               input_names, output_names,               
               start_date,
               end_date = None,
               date_name = 'date',
               apiKey = None,
               datetime_unit = 'D', date_format = "%Y-%m-%d", parsing = 'day', interval = '1d',
               combine_stock_features = True,
               log_prices = False,
               transforms = None,
               train_val_test_periods = None,
               pct_train_val_test = [1., 0., 0.],
               batch_size = -1,
               input_len = [1], output_len = [1],
               shift = [0], stride = 1,
               dt = 1, # day
               pad_data = False,
               print_summary = True,
               device = 'cpu', dtype = torch.float32):

    super(StockDataModule, self).__init__()

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])
          
    self.end_date = self.end_date or date.now().strftime(self.date_format)

    if isinstance(self.start_date, pd._libs.tslibs.timestamps.Timestamp):
      self.start_date = self.start_date.strftime(self.date_format)
    elif isinstance(start_date, datetime):
      self.start_date = self.start_date.strftime(self.date_format)

    if isinstance(self.end_date, pd._libs.tslibs.timestamps.Timestamp):
      self.end_date = self.end_date.strftime(self.date_format)
    elif isinstance(end_date, datetime):
      self.end_date = self.end_date.strftime(self.date_format)

    if self.source == 'yfinance':
      import yfinance
    else:
      import requests

    self.max_input_len, self.max_output_len = np.max(input_len).item(), np.max(output_len).item()
    self.max_shift = np.max(self.shift).item()
    self.start_step = np.max([0, (self.max_input_len - self.max_output_len + self.max_shift)]).item()
    self.predicting = False
                 
  def prepare_data(self):

    if not self.predicting:

      self.input_output_names = np.unique(self.input_names + self.output_names).tolist()

      self.symbols = np.unique([name.split('_')[0] for name in self.input_output_names]).tolist()

      ## Download data
      if self.source == 'polygon':
        df = load_polygon(apiKey = self.apiKey,
                          symbols = self.symbols,
                          start_date = self.start_date,
                          end_date = self.end_date,
                          parsing = self.parsing,
                          # datetime_unit = self.datetime_unit,
                          date_format = self.date_format)

      elif self.source == 'yfinance':       
        df = load_yfinance(symbols = self.symbols,
                          start_date = self.start_date,
                          end_date = self.end_date,
                          interval = self.interval,
                          # datetime_unit = self.datetime_unit,
                          date_format = self.date_format)

      else: # if the source is a csv file
        df = pd.read_csv(self.source, usecols = self.input_output_names)

      # log prices if desired
      if self.log_prices:
        for col in df.columns:
          if any(keyword in col for keyword in ['open', 'close', 'high', 'low']):
            df[col] = np.log(df[col])
      #

      # Get volatility, if desired.
      for feature in self.input_output_names:
        feature_split = feature.rsplit('_', 1)

        if 'hv' in feature_split[-1]:
          attr, hvM = feature_split
          months = int(hvM[2:])
          df['_'.join([attr,hvM])] = historical_volatility(df[attr], months = months)
        elif 'dv' in feature_split[-1]:
          attr, hvD = feature_split
          days = int(hvD[2:])
          df['_'.join([attr,hvD])] = daily_volatility(df[attr], days = days, interval = self.interval)
      ##

      df = df.filter(items=[self.date_name] + self.input_output_names)

      if np.any(df.isna()):
        df = df[(np.where(df.isna().any(axis = 1))[0].max()+1):]

      # total length of data
      self.data_len = df.shape[0]
      #

      # Convert dataframe to dictionary of tensors. Concatenate stock features, if desired.
      data = {self.date_name: pd.to_datetime(df[self.date_name]).dt.to_period(self.datetime_unit).dt.to_timestamp().values}

      for col in df.columns:
        if col != self.date_name:
          try: data[col] = torch.tensor(df[col].values.reshape(df[col].shape[0], -1)).to(device = self.device,
                                                                                        dtype = self.dtype)
          except: data[col] = df[col].values.reshape(df[col].shape[0], -1).to(device = self.device,
                                                                            dtype = self.dtype)

      self.output_feature_names = {}
      self.output_feature_idx = {}
      s, f = 0, 0
      for symbol in self.symbols:
        stock_features = []
        self.output_feature_names[symbol] = []
        self.output_feature_idx[symbol] = []
        for feature in list(data):
          if symbol in feature:
            stock_features.append(feature)
            self.output_feature_names[symbol].append(feature.replace(f"{symbol}_", ''))
            self.output_feature_idx[symbol].append(s)
            s += 1

        if self.combine_stock_features:
          data[symbol] = torch.cat([data[feature] for feature in stock_features], -1)
          self.output_feature_idx[symbol] = np.arange(f, f+data[symbol].shape[-1])
          f += data[symbol].shape[-1]

          for feature in stock_features: del data[feature]

      if self.combine_stock_features:
        self.input_names, self.output_names = self.symbols, self.symbols
      #

      # set inputs and outputs
      self.input_output_names = np.unique(self.input_names + self.output_names).tolist()

      self.num_inputs, self.num_outputs = len(self.input_names), len(self.output_names)

      self.input_size = [data[feature].shape[-1] for feature in self.input_names]
      self.output_size = [data[feature].shape[-1] for feature in self.output_names]

      self.max_input_size, self.max_output_size = np.max(self.input_size), np.max(self.output_size)
      self.total_input_size, self.total_output_size = np.sum(self.input_size), np.sum(self.output_size)

      if len(self.input_len) == 1: self.input_len = self.input_len*self.num_inputs

      if len(self.output_len) == 1: self.output_len = self.output_len*self.num_outputs
      if len(self.shift) == 1: self.shift = self.shift*self.num_outputs

      self.has_ar = np.isin(self.output_names, self.input_names).any()
      #

      # get indices for autoregression
      j = 0
      output_input_idx = []
      for i,name in enumerate(self.input_names):

        input_idx = torch.arange(j, (j+self.input_size[i])).to(dtype = torch.long)
        if name in self.output_names:
          output_input_idx.append(input_idx)
        j += self.input_size[i]
      output_input_idx = torch.cat(output_input_idx, -1) if len(output_input_idx) > 0 else []

      j = 0
      input_output_idx = []
      for i,name in enumerate(self.output_names):
        size_i = self.output_size[i] if np.sum(self.output_size) > 0 else self.model.hidden_out_features[i] if np.sum(self.model.hidden_out_features) > 0 else self.model.base_hidden_size[i]

        output_idx = torch.arange(j, (j+size_i)).to(dtype = torch.long)
        if name in self.input_names:
          input_output_idx.append(output_idx)
        j += size_i
      input_output_idx = torch.cat(input_output_idx, -1) if len(input_output_idx) > 0 else []

      self.input_output_idx, self.output_input_idx = input_output_idx, output_input_idx
      #

      # set data scalers
      for name in self.input_output_names:
        if self.transforms is None:
          if 'all' in [name for name in self.transforms]:
            self.transforms[name] = self.transforms['all']
          else:
            self.transforms = {name: FeatureTransform(scale_type = 'identity')}
        if name not in self.transforms:
          if 'all' in [name for name in self.transforms]:
            self.transforms[name] = self.transforms['all']
          else:
            self.transforms = {name: FeatureTransform(scale_type = 'identity')}
      #

      #
      num_samples = data[self.input_output_names[0]].shape[0]
      self.num_samples = num_samples
      #

      # transform data
      for name in list(data):
        if (name in list(self.transforms)):
          data[name] = self.transforms[name].fit_transform(data[name])
      #

      data['steps'] = torch.arange(num_samples).to(device = self.device,
                                                   dtype = torch.long)

      self.data = data

  def setup(self,
            stage = None):

    if (stage == 'fit') & (not self.predicting):

      if self.train_val_test_periods is not None:
        train_period = [pd.Period(date_str, freq = self.datetime_unit).to_timestamp() for date_str in self.train_val_test_periods[0]]
        train_data = {name: self.data[name][(self.data[self.date_name]>=train_period[0]) & (self.data[self.date_name]<=train_period[1])] for name in list(self.data)}

        train_len = train_data[self.date_name].shape[0]

        if len(self.train_val_test_periods[1]) > 0:
          val_period = [pd.Period(date_str, freq = self.datetime_unit).to_timestamp() for date_str in self.train_val_test_periods[1]]
          val_data = {name: self.data[name][(self.data[self.date_name]>=val_period[0]) & (self.data[self.date_name]<=val_period[1])] for name in list(self.data)}
          val_len = val_data[self.date_name].shape[0]
        else:
          val_data = {}

        if len(self.train_val_test_periods[2]) > 0:
          test_period = [pd.Period(date_str, freq = self.datetime_unit).to_timestamp() for date_str in self.train_val_test_periods[2]]
          test_data = {name: self.data[name][(self.data[self.date_name]>=test_period[0]) & (self.data[self.date_name]<=test_period[1])] for name in list(self.data)}
          test_len = test_data[self.date_name].shape[0]
        else:
          test_data = {}

      else:
        # Split the data
        train_len = int(self.pct_train_val_test[0] * self.data_len)
        val_len = int(self.pct_train_val_test[1] * self.data_len)

        train_data = {name: self.data[name][:train_len] for name in list(self.data)}
        if self.pct_train_val_test[1] > 0:
          val_data = {name: self.data[name][train_len:(train_len + val_len)] for name in list(self.data)}
        else:
          val_data = {}

        if self.pct_train_val_test[2] > 0:
          test_data = {name: self.data[name][(train_len + val_len):] for name in list(self.data)}
          test_len = len(next(iter(test_data.values())))
        else:
          test_data = {}
          test_len = 0
        #

      self.train_len, self.val_len, self.test_len = train_len, val_len, test_len

      # prepare init_input
      train_init_input, val_init_input, test_init_input  = None, None, None
      #

      ## Pad data
      if self.pad_data and (self.start_step > 0):

        pad_size = self.start_step

        # extend train steps and pad train data
        train_data['steps'] = torch.cat((train_data['steps'],
                                         torch.arange(1,1+pad_size).to(device = self.device,
                                                                       dtype = torch.long) + train_data['steps'][-1]), 0)

        for name in self.input_output_names:
          train_data[name] = torch.nn.functional.pad(train_data[name], (0, 0, pad_size, 0), mode = 'constant', value = 0)
        #

        # pad val data with train data and get val_init
        if len(val_data) > 0:
          val_data['steps'] = torch.cat((train_data['steps'][-pad_size:],
                                          torch.arange(1,1+len(val_data['steps']))+train_data['steps'][-1]))
          for name in self.input_output_names:
            val_data[name] = torch.cat((train_data[name][-pad_size:], val_data[name]), 0)

          val_init_input = val_init_input or []
          for i,name in enumerate(self.input_names):
            val_init_input.append(train_data[name][-(pad_size+1)])
        #

        # pad test data with val (or train) data and get test_init
        if len(test_data) > 0:
          data_ = val_data if len(val_data) > 0 else train_data
          test_data['steps'] = torch.cat((data_['steps'][-pad_size:],
                                          torch.arange(1,1+len(test_data['steps']))+data_['steps'][-1]))
          for name in self.input_output_names:
            test_data[name] = torch.cat((data_[name][-pad_size:], test_data[name]), 0)

          test_init_input = test_init_input or []
          for i,name in enumerate(self.input_names):
            test_init_input.append(data_[name][-(pad_size+1)])
        #
      else: # don't pad data, just get init_inputs for val and test data

        # val or train data for test_init_input
        data_ = val_data if len(val_data) > 0 else train_data
        #

        if (len(val_data) > 0) & self.has_ar: val_init_input = []
        if (len(test_data) > 0) & self.has_ar: test_init_input = []

        for i,name in enumerate(self.input_names):

          # for val
          if (len(val_data) > 0) & self.has_ar:
            val_init_input.append(train_data[name][-1])
          #

          # for test
          if (len(test_data) > 0) & self.has_ar:
            test_init_input.append(data_[name][-1])
          #

      if val_init_input is not None: val_init_input = torch.cat(val_init_input, -1)
      if test_init_input is not None: test_init_input = torch.cat(test_init_input, -1)

      self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
      self.train_init_input, self.val_init_input, self.test_init_input = train_init_input, val_init_input, test_init_input

  ## Setup training dataloader
  def train_dataloader(self):
    if not self.predicting:
      self.train_batch_size = len(self.train_data['steps']) if self.batch_size == -1 else self.batch_size

      self.train_dl = SequenceDataloader(input_names = self.input_names,
                                         output_names = self.output_names,
                                         step_name = 'steps',
                                         data = self.train_data,
                                         batch_size = self.train_batch_size,
                                         input_len = self.input_len, output_len = self.output_len, shift = self.shift, stride = self.stride,
                                         init_input = self.train_init_input,
                                         print_summary = self.print_summary,
                                         device = self.device, dtype = self.dtype)

      self.num_train_batches = self.train_dl.num_batches

      self.train_output_mask = self.train_dl.output_mask
      self.train_input_window_idx, self.train_output_window_idx = self.train_dl.input_window_idx, self.train_dl.output_window_idx
      self.train_total_input_len, self.train_total_output_len = self.train_dl.total_input_len, self.train_dl.total_output_len

      self.train_unique_output_window_idx = self.train_dl.unique_output_window_idx

      print("Training Dataloader Created.")

      return self.train_dl.dl
    else:
      return None
    ##

  ## Setup Validation dataloader
  def val_dataloader(self):
    if not self.predicting:
      if (len(self.val_data) > 0):
        self.val_batch_size = len(self.val_data['steps']) if self.batch_size == -1 else self.batch_size
      else:
        self.val_batch_size = 1

      self.val_dl = SequenceDataloader(input_names = self.input_names,
                                       output_names = self.output_names,
                                       step_name = 'steps',
                                       data = self.val_data,
                                       batch_size = self.val_batch_size,
                                       input_len = self.input_len, output_len = self.output_len, shift = self.shift, stride = self.stride,
                                       init_input = self.val_init_input,
                                       print_summary = self.print_summary,
                                       device = self.device, dtype = self.dtype)

      if len(self.val_dl.dl) > 0:
        self.num_val_batches = self.val_dl.num_batches

        self.val_output_mask = self.val_dl.output_mask
        self.val_input_window_idx, self.val_output_window_idx = self.val_dl.input_window_idx, self.val_dl.output_window_idx
        self.val_total_input_len, self.val_total_output_len = self.val_dl.total_input_len, self.val_dl.total_output_len

        self.val_unique_output_window_idx = self.val_dl.unique_output_window_idx
      else:
        self.num_val_batches = 0

        self.val_output_mask = None
        self.val_input_window_idx, self.val_output_window_idx = None, None
        self.val_total_input_len, self.val_total_output_len = None, None

        self.val_unique_output_window_idx = None

      return self.val_dl.dl

    else:
      return None
  ##

  ## Setup Test dataloader
  def test_dataloader(self):
    if self.predicting & ~hasattr(self, 'test_dl'):

      if (len(self.test_data) > 0):
        self.test_batch_size = len(self.test_data['steps']) if self.batch_size == -1 else self.batch_size
      else:
        self.test_batch_size = 1

      self.test_dl = SequenceDataloader(input_names = self.input_names,
                                      output_names = self.output_names,
                                      step_name = 'steps',
                                      data = self.test_data,
                                      batch_size = self.test_batch_size,
                                      input_len = self.input_len, output_len = self.output_len, shift = self.shift, stride = self.stride,
                                      init_input = self.test_init_input,
                                      print_summary = self.print_summary,
                                      device = self.device, dtype = self.dtype)

      if len(self.test_dl.dl) > 0:
        self.num_test_batches = self.test_dl.num_batches

        self.test_output_mask = self.test_dl.output_mask
        self.test_input_window_idx, self.test_output_window_idx = self.test_dl.input_window_idx, self.test_dl.output_window_idx
        self.test_total_input_len, self.test_total_output_len = self.test_dl.total_input_len, self.test_dl.total_output_len

        self.test_unique_output_window_idx = self.test_dl.unique_output_window_idx
      else:
        self.num_test_batches = 0

        self.test_output_mask = None
        self.test_input_window_idx, self.test_output_window_idx = None, None
        self.test_total_input_len, self.test_total_output_len = None, None

        self.test_unique_output_window_idx = None

      return self.test_dl.dl

    else:
      return None
  ##
