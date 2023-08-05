import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import pickle

# from ts_src.SequenceDataloader import SequenceDataloader
from ts_src.FeatureTransform import FeatureTransform

from datetime import datetime, timedelta

import copy 

class DataModule(pl.LightningDataModule):
  def __init__(self,
                data,
                time_name, input_names, output_names,
                time_shifts = None,
                combine_features = None, transforms = {'all': FeatureTransform('identity')},
                pct_train_val_test = [1., 0., 0.],
                batch_size = -1,
                input_len = [1], output_len = [1], shift = [0], stride = 1,
                dt = None,
                time_unit = 's',
                pad_data = False,
                shuffle_train_batch = False,
                print_summary = True,
                device = 'cpu', dtype = torch.float32):

      '''
      Initializes a DataModule object.

      Args:
          data (str or pd.DataFrame): Path to structured data or a pandas DataFrame containing the data.
          time_name (str): Name of the column in the data that represents time.
          input_names (list): List of input feature names.
          output_names (list): List of output feature names.
          combine_features (list, optional): List of features to fuse into a single feature. Defaults to None.
          transforms (dict, optional): Dictionary specifying the transformations to be applied to each feature. Defaults to None.
          pct_train_val_test (list, optional): List specifying the percentage of data to use for training,
                                                validation, and testing, respectively. Defaults to [1., 0., 0.].
          batch_size (int, optional): Batch size for the dataloaders. If -1, the entire dataset is treated as a
                                      single batch. Defaults to -1.
          input_len (list, optional): List of input sequence lengths for each input feature. If a single value is
                                      provided, it is used for all input features. Defaults to [1].
          output_len (list, optional): List of output sequence lengths for each output feature. If a single value
                                        is provided, it is used for all output features. Defaults to [1].
          shift (list, optional): List of output sequence shifts for each output feature. If a single value is
                                  provided, it is used for all output features. Defaults to [0].
          stride (int, optional): Stride value for creating input-output pairs. Defaults to 1.
          dt (int, optional): Time step size. Defaults to 1.
          time_unit (str, optional): Time unit of the data. Defaults to 's'.
          pad_data (bool, optional): Whether to pad the data to ensure each output sequence has at least one input sequence. Defaults to False.
          print_summary (bool, optional): Whether to print a summary of the data module configuration. Defaults to True.
          device (str, optional): Device to use for tensor operations. Defaults to 'cpu'.
          dtype (torch.dtype, optional): Data type of the tensors. Defaults to torch.float32.
      '''

      super().__init__()

      locals_ = locals().copy()                   
      for arg in locals_:
        if arg != 'self':
          setattr(self, arg, locals_[arg].copy() if arg == 'data' else locals_[arg])  
          
      self.max_input_len = np.max(input_len).item()
      self.max_output_len = np.max(output_len).item()
      self.max_shift = np.max(shift).item()
      self.start_step = np.max([0, (self.max_input_len - self.max_output_len + self.max_shift)]).item()

      self.dt = self.dt or data[time_name].diff().mean()

      self.predicting, self.data_prepared = False, False

  def prepare_data(self):
    '''
    Prepares the data for training, validation, and testing.

    This method is responsible for converting the input data to a dictionary of tensors, applying transformations
    to the data, splitting the data into training, validation, and testing sets, and padding the data if necessary.
    '''
    if not (self.predicting or self.data_prepared):
      self.input_output_names = np.unique(self.input_names + self.output_names).tolist()

      if isinstance(self.data, str):
          # If data is a string, assume it is a path to structured data
          with open(self.data, "rb") as file:
            self.data = pickle.load(file)

      if isinstance(self.data, pd.DataFrame):
          # If data is a pandas dataframe, assume each column is an individual feature
          self.data = self.data.filter(items=[self.time_name] + self.input_output_names)

      # Convert dataframe to dictionary of tensors. Concatenate features, if desired.
      data = {self.time_name: self.data[self.time_name]}
      for name in self.input_output_names:
        if not isinstance(self.data[name], torch.Tensor):
            data[name] = torch.tensor(np.array(self.data[name])).to(device=self.device, dtype=self.dtype)
        else:
            data[name] = self.data[name].to(device=self.device, dtype=self.dtype)

        data[name] = data[name].unsqueeze(1) if data[name].ndim == 1 else data[name]
      
      self.data = data.copy()
      
      mask = np.ones((self.data[self.time_name].shape[0]), dtype = bool)
      
      # Shift data
      if self.time_shifts is not None:
        for name in self.time_shifts:
        
          s = self.time_shifts[name]
  
          self.data[name] = torch.roll(self.data[name], shifts = s, dims = 0)
          
          nan_idx = (torch.arange(s) if s >= 0 else torch.arange(self.data[name].shape[0]+s, self.data[name].shape[0])).to(device = self.device, 
                                                                                                                           dtype = torch.long)
          
          mask[nan_idx.cpu()] = False
  
          self.data[name].index_fill_(0, nan_idx, float('nan'))
  
        if isinstance(self.data[self.time_name], pd.core.series.Series):      
          self.data[self.time_name] = self.data[self.time_name][mask] 
        else:
          self.data[self.time_name] = self.data[self.time_name][mask] 
  
        for name in self.input_output_names:
          self.data[name] = self.data[name][mask]
      #

      self.transforms = {'all': FeatureTransform(transform_type='identity')} if self.transforms is None else self.transforms
      for name in self.input_output_names:
        if 'all' in self.transforms:
          self.transforms[name] = copy.deepcopy(self.transforms['all']) # FeatureTransform(transform_type = self.transforms['all'].transform_type)
        elif name not in self.transforms:
          self.transforms[name] = FeatureTransform(transform_type = 'identity')
        
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
        
        # for name in list(np.unique(self.input_names_original + self.output_names_original)): 
        #   del self.data[name]

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
      
      # Split the data
      train_len = int(self.pct_train_val_test[0] * self.data_len)
      val_len = int(self.pct_train_val_test[1] * self.data_len)

      train_data = {name: self.data[name][:train_len] for name in ([self.time_name, 'steps'] + self.input_output_names)}    
      if self.pct_train_val_test[1] > 0:
        val_data = {name: self.data[name][train_len:(train_len + val_len)] for name in ([self.time_name, 'steps'] + self.input_output_names)}
      else:
        val_data = {}
      
      if self.pct_train_val_test[2] > 0:
        test_data = {name: self.data[name][(train_len + val_len):] for name in ([self.time_name, 'steps'] + self.input_output_names)}
        test_len = len(next(iter(test_data.values())))
      else:
          test_data = {}
          test_len = 0

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
                                          shuffle_batch = self.shuffle_train_batch
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
                                        shuffle_batch = self.shuffle_train_batch,
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
