import torch
import numpy as np

class SequenceDataset(torch.utils.data.Dataset):

  '''
  Dataset class for sequence data.

  Args:
    data (dict): Dictionary containing input and output data.
    input_names (list): Names of the input data.
    output_names (list): Names of the output data.
    step_name (str): Name of the step data.
    input_len (list): List of input sequence lengths. If a single value is provided, it is replicated for all inputs.
    output_len (list): List of output sequence lengths. If a single value is provided, it is replicated for all outputs.
    shift (list): List of output shifts. If a single value is provided, it is replicated for all outputs.
    stride (int): Stride value.
    init_input (torch.Tensor or None): Initial input for padding. Defaults to None.
    print_summary (bool): Whether to print summary information. Defaults to False.
    device (str): Device on which the dataset is allocated. Defaults to 'cpu'.
    dtype (torch.dtype): Data type of the dataset. Defaults to torch.float32.
    forecast (bool): Whether the dataset is for forecasting. Defaults to False.
  '''

  def __init__(self,
               data: dict,
               input_names, output_names, step_name='steps',
               input_len=[1], output_len=[1], shift=[0], stride=1,
               init_input=None,
              #  shuffle_batch = False,
               forecast = False,
               print_summary=False,
               device='cpu', dtype=torch.float32):

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        if arg == 'data':
          setattr(self, arg, locals_[arg].copy())
        else:
          setattr(self, arg, locals_[arg])

    self.num_inputs, self.num_outputs = len(self.input_names), len(self.output_names)

    if len(self.input_len) == 1:
        self.input_len = self.input_len * self.num_inputs

    if len(self.output_len) == 1:
        self.output_len = self.output_len * self.num_outputs
    if len(self.shift) == 1:
        self.shift = self.shift * self.num_outputs
    
    for name in self.input_names + self.output_names:
      if not isinstance(self.data[name], torch.Tensor):
        self.data[name] = torch.tensor(self.data[name]).to(device = self.device,
                                                           dtype = self.dtype)

    self.data_len = self.data[self.input_names[0]].shape[0]

    if step_name not in data: self.data[step_name] = torch.arange(self.data_len).to(device = self.device,
                                                                                    dtype = torch.long)

    self.input_len = [self.data_len if len == -1 else len for len in self.input_len]
    self.output_len = [np.max(self.input_len) if len == -1 else len for len in self.output_len]

    self.input_size = [self.data[name].shape[-1] for name in self.input_names]
    self.output_size = [self.data[name].shape[-1] for name in self.output_names]

    self.total_input_len = len(torch.cat(self.input_window_idx).unique())
    self.total_output_len = len(torch.cat(self.output_window_idx).unique())
    self.max_shift = np.max(self.shift)

    self.has_ar = np.isin(self.output_names, self.input_names).any()

    self.input_window_idx = []
    for i in range(self.num_inputs):
      self.input_window_idx.append(torch.arange(self.total_input_len - self.input_len[i], self.total_input_len).to(device = 'cpu',
                                                                                                               dtype = torch.long))
      if self.has_ar and (self.input_names[i] not in self.output_names):
        self.input_window_idx[i] += 1

    self.output_window_idx = []
    for i in range(self.num_outputs):
      output_window_idx_i = torch.arange(self.total_input_len - self.output_len[i], self.total_input_len).to(device = 'cpu',
                                                                                                         dtype = torch.long) + self.shift[i]
      self.output_window_idx.append(output_window_idx_i)

      if self.has_ar:
        self.output_window_idx[i] += 1

    self.total_window_size = torch.cat(self.output_window_idx).max().item() + 1
    self.total_window_idx = torch.arange(self.total_window_size).to(device = 'cpu',
                                                                    dtype = torch.long)

    self.start_step = np.max([0, (self.total_input_len - self.total_output_len + self.max_shift + int(self.has_ar))]).item()

    if self.print_summary:
      print('\n'.join([f'Data length: {self.data_len}',
                       f'Window size: {self.total_window_size}',
                       f'Step indices: {self.total_window_idx.tolist()}',
                       '\n'.join([f'Input indices for {self.input_names[i]}: {self.input_window_idx[i].tolist()}' for i in range(self.num_inputs)]),
                       '\n'.join([f'Output indices for {self.output_names[i]}: {self.output_window_idx[i].tolist()}' for i in range(self.num_outputs)])]))

    if self.forecast:
      pad_size = self.total_window_size - self.total_input_len # + int(self.has_ar)

      self.data[self.step_name] = torch.cat((self.data[self.step_name][-self.total_input_len:],
                                             torch.arange(pad_size).to(device = self.device, dtype = torch.long) + self.data[self.step_name].max() + 1)).to(device = self.device,
                                                                                                                                                            dtype = torch.long)
      for name in np.unique(self.input_names + self.output_names):
        data_size = self.data[name].shape[-1]
        self.data[name] = self.data[name][-self.total_input_len:]
        self.data[name] = torch.nn.functional.pad(self.data[name],
                                                  pad = (0, 0, 0, pad_size),
                                                  mode = 'constant', value = 0.)

      self.data_len = len(self.data[self.step_name])

    self.input_samples, self.output_samples, self.steps_samples, self.id = self.get_samples()

  def get_samples(self):

    '''
    Generates input, output, and steps samples for the dataset.

    Returns:
        tuple: A tuple containing input samples, output samples, and steps samples.
    '''

    input_samples, output_samples, steps_samples = [], [], []

    unique_input_window_idx = torch.cat(self.input_window_idx).unique()
    unique_output_window_idx = torch.cat(self.output_window_idx).unique()

    min_output_idx = torch.cat(self.output_window_idx).min().item()

    window_idx_n = self.total_window_idx

    num_samples = 0
    while window_idx_n.max() < self.data_len:
        num_samples += 1

        steps_samples.append(self.data[self.step_name][window_idx_n])

        # input
        input_n = torch.zeros((self.total_input_len, np.sum(self.input_size))).to(device=self.device,
                                                                           dtype=self.dtype)

        j = 0
        for i in range(self.num_inputs):
          input_window_idx_i = self.input_window_idx[i]

          input_samples_window_idx_i = window_idx_n[input_window_idx_i] # - int(self.input_names[i] in self.output_names)

          if (input_samples_window_idx_i[0] == 0) & (self.init_input is not None):
            input_n[0, j:(j + self.input_size[i])] = self.init_input[j:(j + self.input_size[i])]

          # input_window_idx_i = input_window_idx_i[input_samples_window_idx_i >= 0]
          # input_samples_window_idx_i = input_samples_window_idx_i[input_samples_window_idx_i >= 0]

          input_n[input_window_idx_i, j:(j + self.input_size[i])] = self.data[self.input_names[i]].clone()[input_samples_window_idx_i]

          j += self.input_size[i]

        input_samples.append(input_n)

        # output
        output_n = torch.zeros((len(unique_output_window_idx), np.sum(self.output_size))).to(device = self.device,
                                                                                             dtype = self.dtype)

        j = 0
        for i in range(self.num_outputs):
          output_window_idx_i = self.output_window_idx[i]

          output_samples_window_idx_i = window_idx_n[output_window_idx_i]

          output_window_idx_j = output_window_idx_i - min_output_idx

          output_n[output_window_idx_j, j:(j + self.output_size[i])] = self.data[self.output_names[i]].clone()[output_samples_window_idx_i]

          j += self.output_size[i]

        output_samples.append(output_n)

        window_idx_n = num_samples * self.stride + self.total_window_idx

    input_samples = torch.stack(input_samples)
    output_samples = torch.stack(output_samples)
    steps_samples = torch.stack(steps_samples)

    if self.forecast:
      input_samples = input_samples[-1:]
      output_samples = output_samples[-1:]
      steps_samples = steps_samples[-1:]
      num_samples = 1

    self.num_samples = num_samples

    # self.batch_shuffle_idx = None
    # if self.shuffle_batch:
    #   self.batch_shuffle_idx = torch.randperm(self.num_samples)
    #   input_samples, output_samples, steps_samples = input_samples[self.batch_shuffle_idx], output_samples[self.batch_shuffle_idx], steps_samples[self.batch_shuffle_idx]

    return input_samples, output_samples, steps_samples, [self.data['id']]*self.num_samples

  def __len__(self):
    '''
    Returns the number of samples in the dataset.

    Returns:
      int: Number of samples in the dataset.
    '''
    return self.num_samples

  def __getitem__(self, idx):
    '''
    Returns a sample from the dataset at the given index.

    Args:
        idx (int): Index of the sample.

    Returns:
        tuple: A tuple containing the input, output, and steps for the sample.
    '''

    return self.input_samples[idx], self.output_samples[idx], self.steps_samples[idx], self.data['id']
