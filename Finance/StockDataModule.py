import torch
import pytorch_lightning as pl
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from TorchTimeSeries.ts_src.FeatureTransform import FeatureTransform
from TorchTimeSeries.ts_src.SequenceDataloader import SequenceDataloader
from TorchTimeSeries.Finance.load_polygon import load_polygon
from TorchTimeSeries.Finance.load_yfinance import load_yfinance 
from TorchTimeSeries.Finance.historical_volatility import historical_volatility
from TorchTimeSeries.Finance.daily_volatility import daily_volatility

class StockDataModule(pl.LightningDataModule):
  def __init__(self,
               source,
               input_names, output_names, 
               start_time,
               end_time = None,
               time_name = 'date',
               apiKey = None,
               time_unit = 'D', date_format = "%Y-%m-%d", parsing = 'day', interval = '1d',
               combine_features = True,
               log_prices = False,
               transforms = {'all': FeatureTransform('identity')},
               train_val_test_periods = None,
               pct_train_val_test = [1., 0., 0.],
               batch_size = -1,
               input_len = [1], output_len = [1],
               shift = [0], stride = 1,
               time_shifts = None,
               dt = None,
               pad_data = False,
               print_summary = True,
               device = 'cpu', dtype = torch.float32):

      super(StockDataModule, self).__init__()

      locals_ = locals().copy()                   
      for arg in locals_:
        if arg != 'self':
          setattr(self, arg, locals_[arg].copy() if arg == 'data' else locals_[arg])  
          
      self.max_input_len = np.max(input_len).item()
      self.max_output_len = np.max(output_len).item()
      self.max_shift = np.max(shift).item()
      self.start_step = np.max([0, (self.max_input_len - self.max_output_len + self.max_shift)]).item()

      if self.interval == '1h':
        self.dt = timedelta(hours = 1)
      else: # self.interval == '1d:
        self.dt = timedelta(days = 1)
                 
      self.predicting, self.data_prepared = False, False

  def prepare_data(self):

    if not (self.predicting or self.data_prepared):

      self.input_output_names = np.unique(self.input_names + self.output_names).tolist()

      self.symbols = np.unique([name.split('_')[0] for name in self.input_output_names]).tolist()

      ## Download data
      if self.source == 'polygon':
        df = load_polygon(apiKey = self.apiKey,
                          symbols = self.symbols,
                          start_time = self.start_time,
                          end_time = self.end_time,
                          parsing = self.parsing,
                          date_format = self.date_format)

      elif self.source == 'yfinance':       
        df = load_yfinance(symbols = self.symbols,
                          start_time = self.start_time,
                          end_time = self.end_time,
                          interval = self.interval,
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

      df = df.filter(items = [self.time_name] + self.input_output_names)

      if np.any(df.isna()):
        df = df[(np.where(df.isna().any(axis = 1))[0].max()+1):]
      
      # total length of data
      self.data_len = df.shape[0]
      #
      
      # Convert dataframe to dictionary of tensors. Concatenate features, if desired.
      data = {self.time_name: df[self.time_name]}
      for name in self.input_output_names:
        data[name] = torch.tensor(np.array(df[name])).to(device = self.device, dtype = self.dtype)
        
        data[name] = data[name].unsqueeze(1) if data[name].ndim == 1 else data[name]
      
      self.data = data.copy()
      
      mask = torch.ones((self.data[self.time_name].shape[0]), dtype = bool)
      
      # Shift data
      if self.time_shifts is not None:
        for name in self.time_shifts:
        
          s = self.time_shifts[name]
  
          self.data[name] = torch.roll(self.data[name], shifts = s, dims = 0)
          
          nan_idx = (torch.arange(s) if s >= 0 else torch.arange(self.data[name].shape[0]+s, self.data[name].shape[0])).to(device = self.device)
          
          mask[nan_idx] = False
  
          self.data[name].index_fill_(0, nan_idx, float('nan'))
  
        if isinstance(self.data[self.time_name], pd.core.series.Series):      
          self.data[self.time_name] = self.data[self.time_name].values[mask] 
        else:
          self.data[self.time_name] = self.data[self.time_name][mask] 
  
        for name in self.input_output_names:
          self.data[name] = self.data[name][mask]
      #

      self.transforms = {'all': FeatureTransform(transform_type='identity')} if self.transforms is not None else self.transforms
      for name in self.input_output_names:
        if 'all' in self.transforms:
          self.transforms[name] = self.transforms['all']
        elif name not in self.transforms:
          self.transforms[name] = FeatureTransform(transform_type='identity')
        
        self.data[name] = self.transforms[name].fit_transform(self.data[name])
                    
      self.input_feature_names, self.output_feature_names = None, None
      if self.combine_features:
        self.input_names_original = self.input_names
        self.data['X'] = torch.cat([self.data[name] for name in self.input_names_original],-1)        
        self.input_names, self.num_inputs = ['X'], 1
        self.input_feature_names = self.input_names_original

        self.output_names_original = self.output_names
        self.data['y'] = torch.cat([self.data[name] for name in self.output_names_original],-1)        
        self.output_names, self.num_outputs = ['y'], 1
        self.output_feature_names = self.output_names_original

      self.input_output_names = np.unique(self.input_names + self.output_names).tolist()
      self.num_inputs, self.num_outputs = len(self.input_names), len(self.output_names)
      self.input_size = [self.data[name].shape[-1] for name in self.input_names]
      self.output_size = [self.data[name].shape[-1] for name in self.output_names]
      self.max_input_size, self.max_output_size = np.max(self.input_size), np.max(self.output_size)
      
      if len(self.input_len) == 1:
          self.input_len = self.input_len * self.num_inputs
      if len(self.output_len) == 1:
          self.output_len = self.output_len * self.num_outputs

      if len(self.shift) == 1:
          self.shift = self.shift * self.num_outputs
      
      self.has_ar = np.isin(self.output_names, self.input_names).any()

      self.data_len = self.data[self.input_output_names[0]].shape[0]

      self.data['steps'] = torch.arange(self.data_len).to(device=self.device, dtype=torch.long)

      j = 0
      output_input_idx = []
      for i, name in enumerate(self.input_names):
          input_idx = torch.arange(j, (j + self.input_size[i])).to(dtype=torch.long)
          if name in self.output_names:
              output_input_idx.append(input_idx)
          j += self.input_size[i]
      output_input_idx = torch.cat(output_input_idx, -1) if len(output_input_idx) > 0 else []

      j = 0
      input_output_idx = []
      for i, name in enumerate(self.output_names):
          size_i =  self.output_size[i] if np.sum(self.output_size) > 0 \
                    else self.model.hidden_out_features[i] if np.sum(self.model.hidden_out_features) > 0 \
                    else self.model.base_hidden_size[i]

          output_idx = torch.arange(j, (j + size_i)).to(dtype=torch.long)
          if name in self.input_names:
              input_output_idx.append(output_idx)
          j += size_i
      input_output_idx = torch.cat(input_output_idx, -1) if len(input_output_idx) > 0 else []

      self.input_output_idx, self.output_input_idx = input_output_idx, output_input_idx
      self.data_prepared = True
      
  def setup(self, stage):
    '''
    Sets up the data module for a specific stage of training.

    Args:
        stage (str, optional): The current stage of training ('fit' or 'predict'). Defaults to None.
    '''
    
    if (stage == 'fit') and (not self.predicting):
      
      if self.train_val_test_periods is not None:
        train_period = [pd.Period(date_str, freq = self.time_unit).to_timestamp() for date_str in self.train_val_test_periods[0]]
        train_start_time = pd.to_datetime(train_period[0]).tz_localize(self.data[self.time_name].dt.tz) 
        train_end_time = pd.to_datetime(train_period[1]).tz_localize(self.data[self.time_name].dt.tz) 
        
        train_data = {name: self.data[name][(self.data[self.time_name] >= train_start_time) & (self.data[self.time_name] <= train_end_time)] for name in list(self.data)}

        train_len = train_data[self.time_name].shape[0]

        if len(self.train_val_test_periods[1]) > 0:
          val_period = [pd.Period(date_str, freq = self.time_unit).to_timestamp() for date_str in self.train_val_test_periods[1]]
          val_start_time = pd.to_datetime(val_period[0]).tz_localize(self.data[self.time_name].dt.tz) 
          val_end_time = pd.to_datetime(val_period[1]).tz_localize(self.data[self.time_name].dt.tz) 
          
          val_data = {name: self.data[name][(self.data[self.time_name] >= val_start_time) & (self.data[self.time_name] <= val_end_time)] for name in list(self.data)}
  
          val_len = val_data[self.time_name].shape[0]
        else:
          val_data = {}

        if len(self.train_val_test_periods[2]) > 0:
          test_period = [pd.Period(date_str, freq = self.time_unit).to_timestamp() for date_str in self.train_val_test_periods[2]]
          test_start_time = pd.to_datetime(test_period[0]).tz_localize(self.data[self.time_name].dt.tz) 
          test_end_time = pd.to_datetime(test_period[1]).tz_localize(self.data[self.time_name].dt.tz) 
          
          test_data = {name: self.data[name][(self.data[self.time_name] >= test_start_time) & (self.data[self.time_name] <= test_end_time)] for name in list(self.data)}
  
          test_len = test_data[self.time_name].shape[0]
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

      train_init_input, val_init_input, test_init_input = None, None, None

      if self.pad_data and (self.start_step > 0):

        pad_dim = self.start_step
        
        train_data['steps'] = torch.cat((train_data['steps'],
                                         torch.arange(1, 1 + pad_dim).to(device=self.device, dtype=torch.long) + train_data['steps'][-1]),0)

        for name in self.input_output_names:
          train_data[name] = torch.nn.functional.pad(train_data[name], (0, 0, pad_dim, 0), mode='constant', value=0)

        if len(val_data) > 0:
          val_data['steps'] = torch.cat((train_data['steps'][-pad_dim:], torch.arange(1, 1 + len(val_data['steps'])).to(train_data['steps']) + train_data['steps'][-1]))
          for name in self.input_output_names:
              val_data[name] = torch.cat((train_data[name][-pad_dim:], val_data[name]), 0)
        
          val_init_input = val_init_input or []
          for i, name in enumerate(self.input_names):
              val_init_input.append(train_data[name][-(pad_dim + 1)])
          
        if len(test_data) > 0:
          data_ = val_data if len(val_data) > 0 else train_data
          test_data['steps'] = torch.cat((data_['steps'][-pad_dim:], torch.arange(1, 1 + len(test_data['steps'])).to(data_['steps']) + data_['steps'][-1]))
          for name in self.input_output_names:
            test_data[name] = torch.cat((data_[name][-pad_dim:], test_data[name]), 0)

          test_init_input = test_init_input or []
          for i, name in enumerate(self.input_names):
            test_init_input.append(data_[name][-(pad_dim + 1)])

        else:

          data_ = val_data if len(val_data) > 0 else train_data

          if (len(val_data) > 0) and self.has_ar:
            val_init_input = []
          if (len(test_data) > 0) and self.has_ar:
            test_init_input = []

          for i, name in enumerate(self.input_names):

              if (len(val_data) > 0) and self.has_ar:
                val_init_input.append(train_data[name][-1])

              if (len(test_data) > 0) and self.has_ar:
                test_init_input.append(data_[name][-1])

        if val_init_input is not None:
          val_init_input = torch.cat(val_init_input, -1)
        if test_init_input is not None:
          test_init_input = torch.cat(test_init_input, -1)
      
      self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
      self.train_init_input, self.val_init_input, self.test_init_input = train_init_input, val_init_input, test_init_input

  def forecast_dataloader(self):
    '''
    Returns the forecast dataloader.

    Returns:
        torch.utils.data.DataLoader: The forecast dataloader.
    '''
    if len(self.test_data) > 0:
      data = self.test_data.copy()
      init_input = self.test_init_input
      input_len = self.test_max_input_len
      output_len = self.test_max_output_len
      self.last_time = self.test_data[self.time_name].max()
      
    elif len(self.val_data) > 0:
      data = self.val_data.copy()
      init_input = self.val_init_input
      input_len = self.val_max_input_len
      output_len = self.val_max_output_len
      self.last_time = self.val_data[self.time_name].max()
    else:
      data = self.train_data.copy()
      init_input = self.train_init_input
      input_len = self.train_max_input_len
      output_len = self.train_max_output_len
      self.last_time = self.train_data[self.time_name].max()
    
    self.forecast_dl = SequenceDataloader(input_names = self.input_names, 
                                          output_names = self.output_names,
                                          step_name = 'steps',
                                          data = data,
                                          batch_size = 1,
                                          input_len = self.input_len, 
                                          output_len = self.output_len,
                                          shift = self.shift,
                                          stride = self.stride,
                                          init_input = init_input,
                                          forecast = True,
                                          print_summary = False,
                                          device = self.device, dtype = self.dtype)
    
    self.forecast_output_mask = self.forecast_dl.output_mask
    self.forecast_input_window_idx, self.forecast_output_window_idx = self.forecast_dl.input_window_idx, self.forecast_dl.output_window_idx
    self.forecast_max_input_len, self.forecast_max_output_len = self.forecast_dl.max_input_len, self.forecast_dl.max_output_len

    self.forecast_unique_output_window_idx = self.forecast_dl.unique_output_window_idx

    return self.forecast_dl.dl
      
  def train_dataloader(self):
    '''
    Returns the training dataloader.

    Returns:
        torch.utils.data.DataLoader: The training dataloader.
    '''
    if not self.predicting:
      self.train_batch_size = len(self.train_data['steps']) if self.batch_size == -1 else self.batch_size

      self.train_dl = SequenceDataloader(input_names=self.input_names,
                                          output_names=self.output_names,
                                          step_name='steps',
                                          data=self.train_data,
                                          batch_size=self.train_batch_size,
                                          input_len=self.input_len,
                                          output_len=self.output_len,
                                          shift=self.shift,
                                          stride=self.stride,
                                          init_input=self.train_init_input,
                                          print_summary=self.print_summary,
                                          device=self.device,
                                          dtype=self.dtype)
      self.num_train_batches = self.train_dl.num_batches

      self.train_output_mask = self.train_dl.output_mask
      self.train_input_window_idx, self.train_output_window_idx = self.train_dl.input_window_idx, self.train_dl.output_window_idx
      self.train_max_input_len, self.train_max_output_len = self.train_dl.max_input_len, self.train_dl.max_output_len

      self.train_unique_output_window_idx = self.train_dl.unique_output_window_idx

      print("Training Dataloader Created.")

      return self.train_dl.dl
    else:
      return None

  def val_dataloader(self):
    '''
    Returns the validation dataloader.

    Returns:
        torch.utils.data.DataLoader: The validation dataloader.
    '''
    if not self.predicting:
      if len(self.val_data) > 0:
        self.val_batch_size = len(self.val_data['steps']) if self.batch_size == -1 else self.batch_size
      else:
        self.val_batch_size = 1

      self.val_dl = SequenceDataloader(input_names=self.input_names,
                                      output_names=self.output_names,
                                      step_name='steps',
                                      data=self.val_data,
                                      batch_size=self.val_batch_size,
                                      input_len=self.input_len,
                                      output_len=self.output_len,
                                      shift=self.shift,
                                      stride=self.stride,
                                      init_input=self.val_init_input,
                                      print_summary=self.print_summary,
                                      device=self.device,
                                      dtype=self.dtype)

      self.num_val_batches = self.val_dl.num_batches

      self.val_output_mask = self.val_dl.output_mask
      self.val_input_window_idx, self.val_output_window_idx = self.val_dl.input_window_idx, self.val_dl.output_window_idx
      self.val_max_input_len, self.val_max_output_len = self.val_dl.max_input_len, self.val_dl.max_output_len

      self.val_unique_output_window_idx = self.val_dl.unique_output_window_idx

      return self.val_dl.dl
    else:
      return None

  def test_dataloader(self):
    '''
    Returns the test dataloader.

    Returns:
        torch.utils.data.DataLoader: The test dataloader.
    '''
    if self.predicting and not hasattr(self, 'test_dl'):
      if len(self.test_data) > 0:
        self.test_batch_size = len(self.test_data['steps']) if self.batch_size == -1 else self.batch_size
      else:
        self.test_batch_size = 1

      self.test_dl = SequenceDataloader(input_names=self.input_names,
                                        output_names=self.output_names,
                                        step_name='steps',
                                        data=self.test_data,
                                        batch_size=self.test_batch_size,
                                        input_len=self.input_len,
                                        output_len=self.output_len,
                                        shift=self.shift,
                                        stride=self.stride,
                                        init_input=self.test_init_input,
                                        print_summary=self.print_summary,
                                        device=self.device,
                                        dtype=self.dtype)

      self.num_test_batches = self.test_dl.num_batches

      self.test_output_mask = self.test_dl.output_mask
      self.test_input_window_idx, self.test_output_window_idx = self.test_dl.input_window_idx, self.test_dl.output_window_idx
      self.test_max_input_len, self.test_max_output_len = self.test_dl.max_input_len, self.test_dl.max_output_len

      self.test_unique_output_window_idx = self.test_dl.unique_output_window_idx

      return self.test_dl.dl
    else:
      return None
