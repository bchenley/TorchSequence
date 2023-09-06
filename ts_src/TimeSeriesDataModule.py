import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd
import pickle

from ts_src.SequenceDataloader import SequenceDataloader
from ts_src.FeatureTransform import FeatureTransform

from datetime import datetime, timedelta

import copy

class TimeSeriesDataModule(pl.LightningDataModule):
  
  def __init__(self,
               data,
               time_name, input_names, output_names,
               step_shifts = None,
               combine_inputs = None, combine_outputs = None,
               transforms = None,
               pct_train_val_test = [1., 0., 0.],
               train_val_test_periods = None,
               batch_size = -1,
               input_len = [1], output_len = [1], shift = [0], stride = 1,
               dt = None,
               time_unit = 's',
               input_unit = [None], output_unit = [None],
               pad_data = False,
               shuffle_train_batch = False,
               print_summary = False,
               device = 'cpu', dtype = torch.float32):

    """
    Initialize the TimeSeriesDataModule.

    Args:
        data (Union[str, List[dict], pd.DataFrame]): The input data.
        time_name (str): Name of the time column.
        input_names (List[str]): Names of input columns.
        output_names (List[str]): Names of output columns.
        step_shifts (Optional[dict]): Shift values for specific columns.
        combine_inputs (Optional[List[List[str]]]): List of input feature names to be combined.
        combine_outputs (Optional[List[List[str]]]): List of output target names to be combined.
        transforms (Optional[dict]): Dictionary of FeatureTransform instances.
        pct_train_val_test (List[float]): Percentage of data for train, validation, and test sets.
        train_val_test_periods (Optional[List[List[str]]]): List of periods for train, validation, and test sets.
        batch_size (int): Batch size for DataLoader.
        input_len (List[int]): Input sequence lengths.
        output_len (List[int]): Output sequence lengths.
        shift (List[int]): Shift values for each output.
        stride (int): Stride value for creating sequences.
        dt (Optional[float]): Time step between data points.
        time_unit (str): Time unit for period-based slicing.
        pad_data (bool): Whether to pad data with NaN values.
        shuffle_train_batch (bool): Whether to shuffle batches during training.
        print_summary (bool): Whether to print data summary.
        device (str): Device for data storage.
        dtype (torch.dtype): Data type for tensors.
    """

    super().__init__()

    locals_ = locals().copy()

    num_inputs, num_outputs = len(input_names), len(output_names)
                 
    for arg in locals_:
      if arg != 'self':
        value = locals_[arg]
        
        if isinstance(value, list) and ('input_' in arg):
          if len(value) == 1:
            setattr(self, arg, value * num_inputs)
          else:
            setattr(self, arg, value)
        elif isinstance(value, list) and ('input_' in arg):
          if len(value) == 1:
            setattr(self, arg, value * num_outputs)
          else:
            setattr(self, arg, value)
        else:
            setattr(self, arg, value)

    self.input_names_original, self.output_names_original = self.input_names, self.output_names

    self.input_output_names = np.unique(self.input_names + self.output_names).tolist()
    self.input_output_names_original = self.input_output_names

    if not isinstance(self.dt, timedelta):
      self.dt = timedelta(seconds = self.dt)
      
    if self.transforms is None:
      self.transforms = {'all': FeatureTransform(transform_type = 'identity')}

    for name in self.input_output_names:
      if 'all' in self.transforms:
        self.transforms[name] = copy.deepcopy(self.transforms['all'])
      elif name not in self.transforms:
        self.transforms[name] = FeatureTransform(transform_type = 'identity')

    if 'all' in self.transforms: del self.transforms['all']

    self.has_ar = np.isin(self.output_names, self.input_names).any()

    self.max_input_len = np.max(input_len).item()
    self.max_output_len = np.max(output_len).item()
    self.max_shift = np.max(shift).item()
    self.start_step = np.max([0, (self.max_input_len - self.max_output_len + self.max_shift + int(self.has_ar)]).item()

    self.predicting, self.data_prepared = False, False

  def prepare_data(self):
    """
    Preprocesses the input data for training, validation, and testing.
    """
    # Check if data has already been prepared or if in prediction mode
    if not (self.predicting or self.data_prepared):

        # Load data from a pickled file if data is a string
        if isinstance(self.data, str):
            with open(self.data, "rb") as file:
                self.data = pickle.load(file)

        # Convert single dataset to a list
        if not isinstance(self.data, list):
            self.data = [self.data]

        # Store the number of datasets
        self.num_datasets = len(self.data)

        # Store information about input and output features
        self.num_inputs = len(self.input_names)
        self.input_size = [self.data[0][name].shape[-1] for name in self.input_names]
        self.input_feature_names = self.input_names
        self.input_feature_size = self.input_size

        self.num_outputs = len(self.output_names)
        self.output_size = [self.data[0][name].shape[-1] for name in self.output_names]
        self.output_feature_names = self.output_names
        self.output_feature_size = self.output_size

        # Initialize variables for indexing input/output features
        j = 0
        output_input_idx = []
        for i, name in enumerate(self.input_names):
            input_idx = torch.arange(j, (j + self.input_size[i])).to(dtype = torch.long)
            if name in self.output_names:
                output_input_idx.append(input_idx)
            j += self.input_size[i]
        output_input_idx = torch.cat(output_input_idx, -1) if len(output_input_idx) > 0 else []

        j = 0
        input_output_idx = []
        for i, name in enumerate(self.output_names):
            size_i = (self.output_size[i]
                      if sum(self.output_size) > 0
                      else self.model.hidden_out_features[i]
                      if sum(self.model.hidden_out_features) > 0
                      else self.model.base_hidden_size[i])

            output_idx = torch.arange(j, (j + size_i)).to(dtype = torch.long)
            if name in self.input_names:
                input_output_idx.append(output_idx)
            j += size_i
        input_output_idx = torch.cat(input_output_idx, -1) if len(input_output_idx) > 0 else []

        self.output_input_idx, self.input_output_idx = output_input_idx, input_output_idx

        # Create copies of transforms for each dataset
        self.transforms = [self.transforms.copy() for _ in range(self.num_datasets)]

        # Store the length of each dataset
        self.data_len = []

        # Loop over each dataset
        for data_idx in range(self.num_datasets):
            # Add an 'id' column to the data if it doesn't exist
            if 'id' not in self.data[data_idx]:
                self.data[data_idx]['id'] = str(data_idx)

            # Convert DataFrame data to a specific format
            if isinstance(self.data[data_idx], pd.DataFrame):
                self.data[data_idx] = self.data.filter(items=[self.time_name] + self.input_output_names_original)

            # Create a dictionary to store the preprocessed data
            data = {self.time_name: self.data[data_idx][self.time_name]}
            data['id'] = self.data[data_idx]['id']

            # Process time index and convert it to timedelta
            if not isinstance(data[self.time_name], pd.Series):
              time_idx = data[self.time_name]
              if not isinstance(time_idx, pd.Series):
                if isinstance(time_idx, torch.Tensor):
                    time_idx = time_idx.cpu().numpy()
                data[self.time_name] = pd.Series(time_idx * self.dt)
            
            # Iterate over input and output feature names
            for name in self.input_output_names_original:
                # Convert non-Tensor data to Tensor
                if not isinstance(self.data[data_idx][name], torch.Tensor):
                    data[name] = torch.tensor(np.array(self.data[data_idx][name])).to(device=self.device, dtype=self.dtype)
                else:
                    data[name] = self.data[data_idx][name].to(device=self.device, dtype=self.dtype)

                # Ensure that the Tensor has a time dimension
                data[name] = data[name].unsqueeze(1) if data[name].ndim == 1 else data[name]

            # Store the preprocessed data
            self.data[data_idx] = data.copy()

            # Apply data shifting
            if self.step_shifts is not None:
              mask = np.ones(len(self.data[data_idx][self.time_name]), dtype=bool)

              for name in self.input_output_names_original:
                if name in self.step_shifts:
                  s = self.step_shifts[name]

                  # Roll the data tensor along the specified dimension
                  self.data[data_idx][name] = torch.roll(self.data[data_idx][name], shifts=s, dims=0)

                  # Create a mask for NaN values introduced by rolling
                  nan_idx = (torch.arange(s) if s >= 0 else torch.arange(self.data[data_idx][name].shape[0] + s, self.data[data_idx][name].shape[0])).to(device=self.device, dtype=torch.long)

                  mask[nan_idx.cpu()] = False

                  # Fill NaN values with float('nan')
                  self.data[data_idx][name].index_fill_(0, nan_idx, float('nan'))

              # Apply the mask to the time index
              if isinstance(self.data[data_idx][self.time_name], pd.core.series.Series):
                self.data[data_idx][self.time_name] = self.data[data_idx][self.time_name][mask]
              else:
                self.data[data_idx][self.time_name] = self.data[data_idx][self.time_name][mask]

              # Apply the mask to other input/output features
              for name in self.input_output_names_original:
                self.data[data_idx][name] = self.data[data_idx][name][mask]

            # Apply feature transformations to input/output features
            for name in self.input_output_names_original:
              self.data[data_idx][name] = self.transforms[data_idx][name].fit_transform(self.data[data_idx][name])

            # Combine input features if specified
            if self.combine_inputs:
                
                inputs_combined = []
                new_input_names = []
                for i, input_names in enumerate(self.combine_inputs):
                    input_name_i = f"X{i+1}"
                    self.data[data_idx][input_name_i] = torch.cat([self.data[data_idx][name] for name in input_names], -1)
                    inputs_combined += input_names
                    new_input_names += [input_name_i]

                old_input_names = [name for name in self.input_names if name not in inputs_combined]

                self.input_names = old_input_names + new_input_names
                self.num_inputs = len(self.input_names)
                self.input_size = [self.data[data_idx][name].shape[-1] for name in self.input_names]

            # Combine output targets if specified
            if self.combine_outputs:
                outputs_combined = []
                new_output_names = []
                for i, output_names in enumerate(self.combine_outputs):
                    output_name_i = f"Y{i+1}"
                    self.data[data_idx][output_name_i] = torch.cat([self.data[data_idx][name] for name in output_names], -1)
                    outputs_combined += output_names
                    new_output_names += [output_name_i]

                old_output_names = [name for name in self.output_names if name not in outputs_combined]

                self.output_names = old_output_names + new_output_names
                self.num_outputs = len(self.output_names)
                self.output_size = [self.data[data_idx][name].shape[-1] for name in self.output_names]

            # Update the list of input/output names
            self.input_output_names = np.unique(self.input_names + self.output_names).tolist()

            # Update single values to lists if necessary
            if len(self.input_len) == 1:
                self.input_len = self.input_len * self.num_inputs
            if len(self.output_len) == 1:
                self.output_len = self.output_len * self.num_outputs

            if len(self.shift) == 1:
                self.shift = self.shift * self.num_outputs

            # Store the length of the data for this dataset
            self.data_len.append(self.data[data_idx][self.input_output_names[0]].shape[0])

            # Create a tensor of step indices
            self.data[data_idx]['steps'] = torch.arange(self.data_len[data_idx]).to(device=self.device, dtype=torch.long)

        # # Initialize variables for indexing input/output features
        # j = 0
        # output_input_idx = []
        # for i, name in enumerate(self.input_names):
        #     input_idx = torch.arange(j, (j + self.input_size[i])).to(dtype = torch.long)
        #     if name in self.output_names:
        #         output_input_idx.append(input_idx)
        #     j += self.input_size[i]
        # output_input_idx = torch.cat(output_input_idx, -1) if len(output_input_idx) > 0 else []

        # j = 0
        # input_output_idx = []
        # for i, name in enumerate(self.output_names):
        #     size_i = (self.output_size[i]
        #               if sum(self.output_size) > 0
        #               else self.model.hidden_out_features[i]
        #               if sum(self.model.hidden_out_features) > 0
        #               else self.model.base_hidden_size[i])

        #     output_idx = torch.arange(j, (j + size_i)).to(dtype = torch.long)
        #     if name in self.input_names:
        #         input_output_idx.append(output_idx)
        #     j += size_i
        # input_output_idx = torch.cat(input_output_idx, -1) if len(input_output_idx) > 0 else []
        # self.input_output_idx, self.output_input_idx = input_output_idx, output_input_idx

        # If there's only one dataset, consolidate data and transforms
        if self.num_datasets == 1:
          self.data = self.data[0]
          self.transforms = self.transforms[0]
          self.data_len = self.data_len[0]

        # Mark data as prepared
        self.data_prepared = True

  def setup(self, stage):
    """
    Sets up the training, validation, and test datasets based on the provided configuration.

    Args:
        stage (str): Current stage of setup ('fit', 'validate', or 'test').
    """
    if (stage == 'fit') and (not self.predicting):
      if isinstance(self.data, list):
        # Split the data into train, validation, and test sets
        train_len = int(self.pct_train_val_test[0] * self.num_datasets)
        val_len = int(self.pct_train_val_test[1] * self.num_datasets)

        train_data = self.data[:train_len]
        val_data = self.data[train_len:(train_len + val_len)]
        test_data = self.data[(train_len + val_len):]
        test_len = len(test_data)

        self.train_len, self.val_len, self.test_len = train_len, val_len, test_len
        train_init_input, val_init_input, test_init_input = None, None, None

      else:

        if self.train_val_test_periods is not None:
          # Split data based on specified time periods
          train_period = [pd.Period(time_str, freq = self.time_unit).to_timestamp() for time_str in self.train_val_test_periods[0]]
          train_start_time = pd.to_datetime(train_period[0]).tz_localize(self.data[self.time_name].dt.tz)
          train_end_time = pd.to_datetime(train_period[1]).tz_localize(self.data[self.time_name].dt.tz)

          train_data = {name: self.data[name][(self.data[self.time_name] >= train_start_time) & (self.data[self.time_name] <= train_end_time)] for name in list(self.data)}
          train_len = train_data[self.time_name].shape[0]

          val_period = [pd.Period(time_str, freq = self.time_unit).to_timestamp() for time_str in self.train_val_test_periods[1]]
          val_start_time = pd.to_datetime(val_period[0]).tz_localize(self.data[self.time_name].dt.tz)
          val_end_time = pd.to_datetime(val_period[1]).tz_localize(self.data[self.time_name].dt.tz)

          val_data = {name: self.data[name][(self.data[self.time_name] >= val_start_time) & (self.data[self.time_name] <= val_end_time)] for name in list(self.data)}
          val_len = val_data[self.time_name].shape[0]

          test_period = [pd.Period(time_str, freq=self.time_unit).to_timestamp() for time_str in self.train_val_test_periods[2]]
          test_start_time = pd.to_datetime(test_period[0]).tz_localize(self.data[self.time_name].dt.tz)
          test_end_time = pd.to_datetime(test_period[1]).tz_localize(self.data[self.time_name].dt.tz)

          test_data = {name: self.data[name][(self.data[self.time_name] >= test_start_time) & (self.data[self.time_name] <= test_end_time)] for name in list(self.data)}
          test_len = test_data[self.time_name].shape[0]
        else:

          # Split data based on specified percentages
          train_len = int(self.pct_train_val_test[0] * self.data_len)
          val_len = int(self.pct_train_val_test[1] * self.data_len)

          train_data = {name: self.data[name][:train_len] for name in ([self.time_name, 'steps'] + self.input_output_names)}
          train_data['id'] = self.data['id']

          if self.pct_train_val_test[1] > 0:
            val_data = {name: self.data[name][train_len:(train_len + val_len)] for name in ([self.time_name, 'steps'] + self.input_output_names)}
            val_data['id'] = self.data['id']
          else:
            val_data, val_len = {}, 0

          if self.pct_train_val_test[2] > 0:
            test_data = {name: self.data[name][(train_len + val_len):] for name in ([self.time_name, 'steps'] + self.input_output_names)}
            test_data['id'] = self.data['id']
            test_len = len(next(iter(test_data.values())))
          else:
            test_data, test_len = {}, 0

        self.train_len, self.val_len, self.test_len = train_len, val_len, test_len
        train_init_input, val_init_input, test_init_input = None, None, None

        if self.pad_data and (self.start_step > 0):

          # train_data['steps'] = torch.cat((train_data['steps'],
          #                                  torch.arange(1, 1 + self.start_step).to(device=self.device, dtype=torch.long) + train_data['steps'][-1]),0)

          # for name in self.input_output_names:
          #   train_data[name] = torch.nn.functional.pad(train_data[name], (0, 0, self.start_step, 0), mode='constant', value=0)

          data_ = val_data if len(val_data) > 0 else train_data

          if len(val_data) > 0:
            val_data['steps'] = torch.cat((train_data['steps'][-self.start_step:], torch.arange(1, 1 + len(val_data['steps'])).to(train_data['steps']) + train_data['steps'][-1]))
            for name in self.input_output_names:
              val_data[name] = torch.cat((train_data[name][-self.start_step:], val_data[name]), 0)
            val_init_input = val_init_input or []
            for i, name in enumerate(self.input_names):
              val_init_input.append(train_data[name][-(self.start_step + 1)])
            val_init_input = torch.cat(val_init_input, -1)

          if len(test_data) > 0:
            data_ = val_data if len(val_data) > 0 else train_data
            test_data['steps'] = torch.cat((data_['steps'][-self.start_step:], torch.arange(1, 1 + len(test_data['steps'])).to(data_['steps']) + data_['steps'][-1]))
            for name in self.input_output_names:
              test_data[name] = torch.cat((data_[name][-self.start_step:], test_data[name]), 0)
            test_init_input = test_init_input or []
            for i, name in enumerate(self.input_names):
              test_init_input.append(data_[name][-(self.start_step + 1)])
            test_init_input = torch.cat(test_init_input, -1)

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

      # Store the train, validation, and test data and initialization inputs
      self.train_data, self.val_data, self.test_data = train_data, val_data, test_data
      self.train_init_input, self.val_init_input, self.test_init_input = train_init_input, val_init_input, test_init_input

  def forecast_dataloader(self, print_summary=False):
    """
    Creates and returns a dataloader for generating forecasts using the test or validation data.

    Args:
        print_summary (bool): Whether to print a summary of the created dataloader.

    Returns:
        DataLoader: Forecast dataloader.
    """
    if not hasattr(self, 'forecast_dl'):
      if len(self.test_data) > 0:
        data = self.test_data.copy()
        init_input = self.test_init_input
        input_len = self.test_max_input_len
        output_len = self.test_max_output_len
        self.last_time = self.test_data[self.time_name].max() if not isinstance(self.test_data, list) else None
      elif len(self.val_data) > 0:
        data = self.val_data.copy()
        init_input = self.val_init_input
        input_len = self.val_max_input_len
        output_len = self.val_max_output_len
        self.last_time = self.val_data[self.time_name].max() if not isinstance(self.val_data, list) else None
      else:
        data = self.train_data.copy()
        init_input = self.train_init_input
        input_len = self.train_max_input_len
        output_len = self.train_max_output_len
        self.last_time = self.train_data[self.time_name].max() if not isinstance(self.train_data, list) else None

      # Create a SequenceDataloader for forecasting
      self.forecast_dl = SequenceDataloader(input_names = self.input_names,
                                            output_names = self.output_names,
                                            step_name = 'steps',
                                            data = data,
                                            batch_size = 1,
                                            input_len = input_len,
                                            output_len = output_len,
                                            shift = self.shift,
                                            stride = self.stride,
                                            init_input = init_input,
                                            forecast = True,
                                            print_summary = False,
                                            device = self.device,
                                            dtype = self.dtype)

      # Store the forecast output mask and window indices
      self.forecast_output_mask = self.forecast_dl.output_mask
      self.forecast_input_window_idx, self.forecast_output_window_idx = self.forecast_dl.input_window_idx, self.forecast_dl.output_window_idx
      self.forecast_max_input_len, self.forecast_max_output_len = self.forecast_dl.max_input_len, self.forecast_dl.max_output_len
      self.forecast_unique_output_window_idx = self.forecast_dl.unique_output_window_idx

      print("Forecast Dataloader Created.")

    return self.forecast_dl.dl

  def train_dataloader(self):
    """
    Creates and returns a dataloader for training data.

    Returns:
        DataLoader: Training dataloader.
    """
    if not self.predicting:
        # Set the training batch size
        self.train_batch_size = self.batch_size

        # Create a SequenceDataloader for training
        self.train_dl = SequenceDataloader(
            input_names=self.input_names,
            output_names=self.output_names,
            step_name='steps',
            data=self.train_data,
            batch_size=self.batch_size,
            input_len=self.input_len,
            output_len=self.output_len,
            shift=self.shift,
            stride=self.stride,
            init_input=self.train_init_input,
            shuffle_batch=self.shuffle_train_batch,
            print_summary=self.print_summary,
            device=self.device,
            dtype=self.dtype
        )

        # Update training batch size
        self.train_batch_size = self.train_dl.batch_size

        # Store information about training batches
        self.num_train_batches = self.train_dl.num_batches
        self.train_batch_shuffle_idx = self.train_dl.batch_shuffle_idx
        self.train_output_mask = self.train_dl.output_mask
        self.train_input_window_idx, self.train_output_window_idx = self.train_dl.input_window_idx, self.train_dl.output_window_idx
        self.train_max_input_len, self.train_max_output_len = self.train_dl.max_input_len, self.train_dl.max_output_len
        self.train_unique_output_window_idx = self.train_dl.unique_output_window_idx

        print("Training Dataloader Created.")

        return self.train_dl.dl
    else:
        return None

  def val_dataloader(self):
    """
    Creates and returns a dataloader for validation data.

    Returns:
        DataLoader: Validation dataloader.
    """
    if not self.predicting:
      # Create a SequenceDataloader for validation
      self.val_dl = SequenceDataloader(input_names=self.input_names,
                                        output_names=self.output_names,
                                        step_name='steps',
                                        data=self.val_data,
                                        batch_size=self.batch_size,
                                        input_len=self.input_len,
                                        output_len=self.output_len,
                                        shift=self.shift,
                                        stride=self.stride,
                                        init_input=self.val_init_input,
                                        print_summary=self.print_summary,
                                        device=self.device,
                                        dtype=self.dtype)

      # Store validation batch size
      self.val_batch_size = self.val_dl.batch_size

      # Store information about validation batches
      self.num_val_batches = self.val_dl.num_batches
      self.val_batch_shuffle_idx = self.val_dl.batch_shuffle_idx
      self.val_output_mask = self.val_dl.output_mask
      self.val_input_window_idx, self.val_output_window_idx = self.val_dl.input_window_idx, self.val_dl.output_window_idx
      self.val_max_input_len, self.val_max_output_len = self.val_dl.max_input_len, self.val_dl.max_output_len
      self.val_unique_output_window_idx = self.val_dl.unique_output_window_idx

      print("Validation Dataloader Created.")

      return self.val_dl.dl
    else:
      return None

  def test_dataloader(self):
    """
    Creates and returns a dataloader for test data.

    Returns:
        DataLoader: Test dataloader.
    """
    if self.predicting and not hasattr(self, 'test_dl'):
        # Create a SequenceDataloader for test data
        self.test_dl = SequenceDataloader(input_names=self.input_names,
                                          output_names=self.output_names,
                                          step_name='steps',
                                          data=self.test_data,
                                          batch_size=self.batch_size,
                                          input_len=self.input_len,
                                          output_len=self.output_len,
                                          shift=self.shift,
                                          stride=self.stride,
                                          init_input=self.test_init_input,
                                          shuffle_batch=self.shuffle_train_batch,
                                          print_summary=self.print_summary,
                                          device=self.device,
                                          dtype=self.dtype)

        # Store test batch size
        self.test_batch_size = self.test_dl.batch_size
        self.num_test_batches = self.test_dl.num_batches
        self.test_batch_shuffle_idx = self.test_dl.batch_shuffle_idx
        self.test_output_mask = self.test_dl.output_mask
        self.test_input_window_idx, self.test_output_window_idx = self.test_dl.input_window_idx, self.test_dl.output_window_idx
        self.test_max_input_len, self.test_max_output_len = self.test_dl.max_input_len, self.test_dl.max_output_len
        self.test_unique_output_window_idx = self.test_dl.unique_output_window_idx

        print("Test Dataloader Created.")

        return self.test_dl.dl
    else:
        return None
