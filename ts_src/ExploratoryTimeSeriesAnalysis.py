import torch
import numpy as np
import matplotlib.pyplot as plt

from ts_src.moving_average import moving_average
from ts_src.fft import fft

class ExploratoryTimeSeriesAnalysis():
  """
  A class for performing exploratory time series analysis.

  Parameters:
    data (dict): Dictionary containing the time series data.
    time_name (str): Name of the time column in the data.
    dt (float): Time step or sampling interval.
    time_unit (str): Unit of time (e.g., 'seconds', 'days').
    device (str, optional): Device to be used for computations (e.g., 'cuda', 'cpu'). Defaults to None.
    dtype (torch.dtype, optional): Data type to be used for torch tensors. Defaults to None.
  """

  def __init__(self,
               data, data_names, time_name, dt, time_unit,
               data_units = None,
               device=None, dtype=None):
    """
    Initialize an instance of ExploratoryTimeSeriesAnalysis.

    Args:
      data (dict): Dictionary containing the time series data.
      time_name (str): Name of the time column in the data.
      dt (float): Time step or sampling interval.
      time_unit (str): Unit of time (e.g., 'seconds', 'days').
      device (str, optional): Device to be used for computations (e.g., 'cuda', 'cpu'). Defaults to None.
      dtype (torch.dtype, optional): Data type to be used for torch tensors. Defaults to None.
    """

    locals_ = locals().copy()
    for arg in locals_:
      setattr(self, arg, locals_[arg])

  def generate_xcorr(self, hann_window_len=[None], demean = [False]):
    """
    Generate cross-correlation for the specified data columns.

    Args:
      hann_window_len (list, optional): List of window lengths for applying Hann window to the data.
                                        If a single value is provided, it will be used for all columns.
                                        Defaults to [None] (no Hann window is applied).

    Returns:
      None
    """

    if len(hann_window_len) == 1:
      hann_window_len = hann_window_len * len(self.data_names)

    if len(demean) == 1:
      demean = demean * len(self.data_names)

    data = self.data.copy()

    self.record_len = data[self.data_names[0]].shape[0]

    data_ = []
    for i, name in enumerate(self.data_names):
      data_.append(torch.tensor(self.data[name]).to(device=self.device, dtype=self.dtype)
                    if not isinstance(self.data[name], torch.Tensor) else self.data[name].to(device=self.device,
                                                                                          dtype=self.dtype))

      data_[-1] = data_[-1] - moving_average(data_[-1], torch.hann_window(hann_window_len[i]).to(data_[-1])) \
          if hann_window_len[i] is not None else data_[-1]

      data_[-1] = data_[-1] - data_[-1].mean(0) if demean[i] else data_[-1]

    data = torch.cat(data_, -1)

    data = data.t().unsqueeze(1)

    xcorr_ = torch.nn.functional.conv1d(data, data, padding=(self.record_len - 1))

    self.xcorr = {}
    for i1, name1 in enumerate(self.data_names):
      for i2, name2 in enumerate(self.data_names):
        self.xcorr[f"{name2}[t]*{name1}[t-lag]"] = xcorr_[i1, i2].flip(0)[-self.record_len:] / self.record_len

    self.lags = np.arange(self.record_len) * self.dt

  def plot_ts(self, domain='time', hann_window_len=[None], demean = [False],
              xlim=None,
              figsize=None, title_size=20, xlabel_size=20, ylabel_size=20, fig_num=1):
    """
    Plot the time series data.

    Args:
      domain (str, optional): The domain of the plot, 'time' or 'frequency'. Defaults to 'time'.
      hann_window_len (list, optional): List of window lengths for applying Hann window to the data.
                                        If a single value is provided, it will be used for all columns.
                                        Defaults to [None] (no Hann window is applied).
      data_units (list, optional): List of units for the data columns. Defaults to None.
      xlim (tuple, optional): Limits for the x-axis. Defaults to None (determined automatically).
      figsize (tuple, optional): Figure size in inches. Defaults to None (determined automatically).
      title_size (int, optional): Font size for the plot titles. Defaults to 20.
      xlabel_size (int, optional): Font size for the x-axis label. Defaults to 20.
      ylabel_size (int, optional): Font size for the y-axis label. Defaults to 20.
      fig_num (int, optional): Figure number. Defaults to 1.

    Returns:
      None
    """

    if len(hann_window_len) == 1:
      hann_window_len = hann_window_len * len(self.data_names)
    if len(demean) == 1:
      demean = demean * len(self.data_names)

    data, input_size = [], []
    for name in self.data_names:
      data_ = self.data[name]
      if not isinstance(data_, torch.Tensor):
        data_ = torch.tensor(data_)
      data_ = data_.to(device = self.device, dtype = self.dtype)

      data.append(data_)
      input_size.append(data_.shape[-1])

    data = torch.cat(data, -1)

    num_data = len(self.data_names)

    self.record_len = data.shape[0]

    if domain == 'time':
      self.time = self.data[self.time_name]
      xaxis = self.time
      xlabel = f"Time [{self.time_unit}]"
    elif domain == 'frequency':

      data_ = []
      for i, name in enumerate(self.data_names):
        data_.append(torch.tensor(self.data[name]).to(device=self.device, dtype=self.dtype)
                      if not isinstance(self.data[name], torch.Tensor) else self.data[name].to(device=self.device,
                                                                                          dtype=self.dtype))

        data_[-1] = data_[-1] - moving_average(data_[-1], torch.hann_window(hann_window_len[i]).to(data_[-1])) \
                    if hann_window_len[i] is not None else data_[-1]

        data_[-1] = data_[-1] - data_[-1].mean(0) if demean[i] else data_[-1]

      data = torch.cat(data_, -1)

      self.freq, data, _ = fft(data, fs=1 / self.dt)
      xaxis = self.freq
      xlabel = f"Frequency [1/{self.time_unit}]"

    fig, ax = plt.subplots(num_data, 1, figsize=figsize, num=fig_num)

    j = 0
    for i, name in enumerate(self.data_names):
      ax_i = ax[i] if num_data > 1 else ax
      ax_i.plot(xaxis, data[:, j:(j + input_size[i])], 'b', label=name)
      ax_i.set_title(name, fontsize=title_size)
      ax_i.legend()
      if i == num_data - 1:
        ax_i.set_xlabel(xlabel, fontsize=xlabel_size)
      if self.data_units is not None:
        ax_i.set_ylabel(self.data_units[i], fontsize=ylabel_size)
      ax_i.set_xlim(xlim)
      ax_i.grid()
      ax_i.legend(loc='upper right')

      j += input_size[i]

    fig.tight_layout()

  def plot_xcorr(self, data_names = None, domain='time', xlim=None, figsize=None,
                  title_size=20, xlabel_size=20, ylabel_size=20, fig_num=1):
      """
      Plot the cross-correlation between different data columns.

      Args:
        domain (str, optional): The domain of the plot, 'time' or 'frequency'. Defaults to 'time'.
        data_units (list, optional): List of units for the cross-correlation values. Defaults to None.
        xlim (tuple, optional): Limits for the x-axis. Defaults to None (determined automatically).
        figsize (tuple, optional): Figure size in inches. Defaults to None (determined automatically).
        title_size (int, optional): Font size for the plot titles. Defaults to 20.
        xlabel_size (int, optional): Font size for the x-axis label. Defaults to 20.
        ylabel_size (int, optional): Font size for the y-axis label. Defaults to 20.
        fig_num (int, optional): Figure number. Defaults to 1.

      Returns:
        None
      """

      data_names = data_names or self.data_names

      fig, ax = plt.subplots(len(self.data_names), len(data_names), figsize=figsize, num=fig_num)

      xcorr_names = list(self.xcorr)

      i = -1

      jy = -1
      for ix in range(int(np.sqrt(len(self.xcorr)))):
        for iy in range(int(np.sqrt(len(self.xcorr)))):
          i += 1
          xcorr_ = self.xcorr[xcorr_names[i]].clone()
          if any(f"{name}[t]" in xcorr_names[i] for name in data_names):

            jy += 1
            ax_ixiy = ax[ix, jy] if len(data_names) > 1 else ax[jy] if len(data_names) > 1 else ax

            if domain == 'frequency':
              xaxis, xcorr_, _ = fft(xcorr_, fs=1 / self.dt)
              xlabel = f"Frequency [1/{self.time_unit}]"
              if self.data_units is not None:
                ylabel = f"{self.data_units[ix]}^2*{self.time_unit}" if ix == iy else f"{self.data_units[ix]}*{self.data_units[iy]}/Hz"
            else:
              xaxis = self.lags
              xlabel = f"Lags [{self.time_unit}]"
              if self.data_units is not None:
                ylabel = f"{self.data_units[ix]}^2" if ix == iy else f"{self.data_units[ix]}*{self.data_units[iy]}"

            ax_ixiy.plot(xaxis, xcorr_, 'b')
            ax_ixiy.set_title(xcorr_names[i], fontsize=title_size)
            if ix == int(np.sqrt(len(self.xcorr))) - 1:
              ax_ixiy.set_xlabel(xlabel, fontsize=xlabel_size)
            ax_ixiy.set_ylabel(ylabel, fontsize=ylabel_size)

            ax_ixiy.set_xlim(xlim)
            ax_ixiy.grid()

      fig.tight_layout()
