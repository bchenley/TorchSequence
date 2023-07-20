import torch
import numpy as np

from ts_src.SequenceDataset import SequenceDataset

class SequenceDataloader(torch.utils.data.Dataset):
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
                forecast = False,
                print_summary=False,
                device='cpu', dtype=torch.float32):
    
    super(SequenceDataloader, self).__init__()

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])

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
                            forecast = self.forecast,
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
