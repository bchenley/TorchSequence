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
  '''

  def __init__(self,
                data: dict,
                input_names, output_names, step_name='steps',
                input_len=[1], output_len=[1], shift=[0], stride=1,
                init_input=None,
                print_summary=False,
                device='cpu', dtype=torch.float32):

    num_inputs, num_outputs = len(input_names), len(output_names)

    if len(input_len) == 1:
        input_len = input_len * num_inputs

    if len(output_len) == 1:
        output_len = output_len * num_outputs
    if len(shift) == 1:
        shift = shift * num_outputs

    data_len = data[input_names[0]].shape[0]

    input_len = [data_len if len == -1 else len for len in input_len]
    output_len = [np.max(input_len) if len == -1 else len for len in output_len]

    input_size = [data[name].shape[-1] for name in input_names]
    output_size = [data[name].shape[-1] for name in output_names]

    max_input_len = np.max(input_len)
    max_output_len = np.max(output_len)
    max_shift = np.max(shift)

    has_ar = np.isin(output_names, input_names).any()

    input_window_idx = []
    for i in range(num_inputs):
      input_window_idx.append(torch.arange(max_input_len - input_len[i], max_input_len).to(device=device,
                                                                                              dtype=torch.long))

    output_window_idx = []
    for i in range(num_outputs):
      output_window_idx_i = torch.arange(max_input_len - output_len[i], max_input_len).to(device=device,
                                                                                          dtype=torch.long) + shift[i]
      output_window_idx.append(output_window_idx_i)

    total_window_size = torch.cat(output_window_idx).max().item() + 1
    total_window_idx = torch.arange(total_window_size).to(device=device, dtype=torch.long)

    start_step = max_input_len - max_output_len + max_shift + int(has_ar)

    if print_summary:
      print('\n'.join([f'Data length: {data_len}',
                        f'Window size: {total_window_size}',
                        f'Step indices: {total_window_idx.tolist()}',
                        '\n'.join([f'Input indices for {input_names[i]}: {input_window_idx[i].tolist()}' for i in
                                  range(num_inputs)]),
                        '\n'.join(
                            [f'Output indices for {output_names[i]}: {output_window_idx[i].tolist()}' for i in
                            range(num_outputs)])]))

    self.data = data
    self.input_names, self.output_names, self.step_name = input_names, output_names, step_name
    self.has_ar = has_ar
    self.data_len = data_len
    self.num_inputs, self.num_outputs = num_inputs, num_outputs
    self.input_size, self.output_size = input_size, output_size
    self.start_step = start_step
    self.shift, self.stride = shift, stride
    self.total_window_size, self.total_window_idx = total_window_size, total_window_idx
    self.input_len, self.input_window_idx = input_len, input_window_idx
    self.output_len, self.output_window_idx = output_len, output_window_idx
    self.init_input = init_input
    self.device, self.dtype = device, dtype

    self.input_samples, self.output_samples, self.steps_samples = self.get_samples()

  def get_samples(self):
    '''
    Generates input, output, and steps samples for the dataset.

    Returns:
        tuple: A tuple containing input samples, output samples, and steps samples.
    '''

    input_samples, output_samples, steps_samples = [], [], []

    unique_input_window_idx = torch.cat(self.input_window_idx).unique()
    unique_output_window_idx = torch.cat(self.output_window_idx).unique()

    max_input_len, max_output_len = np.max(self.input_len), np.max(self.output_len + self.shift)

    min_output_idx = torch.cat(self.output_window_idx).min().item()

    window_idx_n = self.total_window_idx

    num_samples = 0
    while window_idx_n.max() < self.data_len:
        num_samples += 1

        steps_samples.append(self.data[self.step_name][window_idx_n])

        # input
        input_n = torch.zeros((max_input_len, np.sum(self.input_size))).to(device=self.device,
                                                                            dtype=self.dtype)

        j = 0
        for i in range(self.num_inputs):
            input_window_idx_i = self.input_window_idx[i]

            input_samples_window_idx_i = window_idx_n[input_window_idx_i] - int(
                self.input_names[i] in self.output_names)

            if (input_samples_window_idx_i[0] == -1) & (self.init_input is not None):
                input_n[0, j:(j + self.input_size[i])] = self.init_input[j:(j + self.input_size[i])]

            input_window_idx_i = input_window_idx_i[input_samples_window_idx_i >= 0]
            input_samples_window_idx_i = input_samples_window_idx_i[input_samples_window_idx_i >= 0]

            input_n[input_window_idx_i, j:(j + self.input_size[i])] = self.data[self.input_names[i]].clone()[
                input_samples_window_idx_i]

            j += self.input_size[i]

        input_samples.append(input_n)

        # output
        output_n = torch.zeros((len(unique_output_window_idx), np.sum(self.output_size))).to(device=self.device,
                                                                                              dtype=self.dtype)

        j = 0
        for i in range(self.num_outputs):
            output_window_idx_i = self.output_window_idx[i]
            output_samples_window_idx_i = window_idx_n[output_window_idx_i]

            output_window_idx_j = output_window_idx_i - min_output_idx

            output_n[output_window_idx_j, j:(j + self.output_size[i])] = self.data[self.output_names[i]].clone()[
                output_samples_window_idx_i]

            j += self.output_size[i]

        output_samples.append(output_n)

        window_idx_n = num_samples * self.stride + self.total_window_idx

    input_samples = torch.stack(input_samples)
    output_samples = torch.stack(output_samples)
    steps_samples = torch.stack(steps_samples)

    self.num_samples = num_samples

    return input_samples, output_samples, steps_samples

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
    return self.input_samples[idx], self.output_samples[idx], self.steps_samples[idx]


class SequenceDataloader:
  '''
  Dataloader class for sequence data.

  Args:
      input_names (list): Names of the input data.
      output_names (list): Names of the output data.
      step_name (str): Name of the step data.
      data (dict): Dictionary containing input and output data.
      batch_size (int): Batch size. Defaults to 1.
      input_len (list): List of input sequence lengths. If a single value is provided, it is replicated for all inputs.
      output_len (list): List of output sequence lengths. If a single value is provided, it is replicated for all outputs.
      shift (list): List of output shifts. If a single value is provided, it is replicated for all outputs.
      stride (int): Stride value. Defaults to 1.
      init_input (torch.Tensor or None): Initial input for padding. Defaults to None.
      print_summary (bool): Whether to print summary information. Defaults to False.
      device (str): Device on which the dataloader is allocated. Defaults to 'cpu'.
      dtype (torch.dtype): Data type of the dataloader. Defaults to torch.float32.
  '''

  def __init__(self,
                input_names, output_names, step_name,
                data: dict,
                batch_size=1,
                input_len=[1], output_len=[1], shift=[0], stride=1,
                init_input=None,
                print_summary=False,
                device='cpu', dtype=torch.float32):

    self.data = data
    self.batch_size = batch_size
    self.input_names, self.output_names, self.step_name = input_names, output_names, step_name
    self.input_len, self.output_len, self.shift, self.stride = input_len, output_len, shift, stride
    self.init_input = init_input
    self.print_summary = print_summary
    self.device, self.dtype = device, dtype

    self.dl = self.get_dataloader

  def collate_fn(self, batch):
    '''
    Collate function for the dataloader.

    Args:
        batch (list): List of samples.

    Returns:
        tuple: A tuple containing input, output, steps, and batch size.
    '''

    input_samples, output_samples, steps_samples = zip(*batch)

    batch_size = len(input_samples)

    pad_fn = lambda x, fill_value: \
        x + tuple(
            torch.full(x[0].shape, fill_value=fill_value).to(device=x[0].device, dtype=x[0].dtype)
            if isinstance(x[0], torch.Tensor)
            else np.full(x[0].shape, fill_value=fill_value)
            for _ in range(self.batch_size - batch_size))

    if batch_size % self.batch_size != 0:
        input_samples = pad_fn(input_samples, 0)
        output_samples = pad_fn(output_samples, 0)
        steps_samples = pad_fn(steps_samples, -1)

    input = torch.stack(input_samples)
    output = torch.stack(output_samples)
    steps = torch.stack(steps_samples)

    return input, output, steps, batch_size

  @property
  def get_dataloader(self):
    '''
    Property function that returns the dataloader.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the sequence dataset.
    '''

    if len(self.data) > 0:
      ds = SequenceDataset(data=self.data,
                            input_names=self.input_names, output_names=self.output_names,
                            step_name=self.step_name,
                            input_len=self.input_len, output_len=self.output_len,
                            shift=self.shift, stride=self.stride,
                            init_input=self.init_input,
                            print_summary=self.print_summary,
                            device=self.device, dtype=self.dtype)

      self.batch_size = len(ds) if self.batch_size == -1 else self.batch_size

      self.input_size, self.output_size = ds.input_size, ds.output_size
      self.num_inputs, self.num_outputs = ds.num_inputs, ds.num_outputs
      self.input_size, self.output_size = ds.input_size, ds.output_size
      self.data_len, self.num_samples = ds.data_len, ds.num_samples
      self.total_window_size, self.total_window_idx = ds.total_window_size, ds.total_window_idx
      self.shift, self.stride = ds.shift, ds.stride
      self.input_len, self.input_window_idx = ds.input_len, ds.input_window_idx
      self.output_len, self.output_window_idx = ds.output_len, ds.output_window_idx

      self.total_input_len, self.total_output_len = len(torch.cat(ds.input_window_idx, 0).unique()), len(
          torch.cat(ds.output_window_idx, 0).unique())
      self.unique_output_window_idx = torch.cat(ds.output_window_idx, 0).unique()

      self.output_mask = torch.zeros((self.total_output_len, np.sum(self.output_size)), device=self.device,
                                      dtype=self.dtype)
      j = 0
      for i in range(len(ds.output_window_idx)):
          output_window_idx_k = [k for k, l in enumerate(self.unique_output_window_idx) if
                                  l in ds.output_window_idx[i]]
          self.output_mask[output_window_idx_k, j:(j + self.output_size[i])] = 1

          j += self.output_size[i]

    else:
      class NoDataset(torch.utils.data.Dataset):
          def __init__(self):
              pass

          def __getitem__(self, index):
              pass

          def __len__(self):
              return 0

      self.input_size, self.output_size = None, None
      self.num_inputs, self.num_outputs = None, None
      self.input_size, self.output_size = None, None
      self.data_len, self.num_samples = None, None
      self.total_window_size, self.total_window_idx = None, None
      self.shift, self.stride = None, None
      self.input_len, self.input_window_idx = None, None
      self.output_len, self.output_window_idx = None, None

      self.total_input_len, self.total_output_len = None, None
      self.unique_output_window_idx = None

      self.output_mask = None

      ds = NoDataset()

    dl = torch.utils.data.DataLoader(ds,
                                      batch_size=self.batch_size,
                                      shuffle=False,
                                      collate_fn=self.collate_fn)

    self.num_batches = len(dl)

    return dl
