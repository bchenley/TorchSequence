import torch
import numpy as np
import matplotlib.pyplot as plt

from ts_src import fft, moving_average

class ExploratoryTimeSeriesAnalysis():
    def __init__(self, 
                 data, time_name, dt, time_unit,
                 device = None, dtype = None):
        """
        Class for exploratory time series analysis.

        Args:
            data (dict): Dictionary containing the time series data.
            time_name (str): Name of the time column in the data.
            dt (float): Time interval between data points.
            time_unit (str): Unit of time.
        """
        locals_ = locals().copy()
        for arg in locals_:
            setattr(self, arg, locals_[arg])

    def generate_xcorr(self, data_names=None, hann_window_len=None):
        """
        Compute cross-correlation between time series data.

        Args:
            data_names (list): Names of the data columns to include in the analysis. Default is None (all columns).
            hann_window_len (int): Length of the Hanning window for data smoothing. Default is None (no smoothing).

        """
        data_names = [name for name in self.data if name != self.time_name] if data_names is None else data_names
        data = self.data.copy()

        self.record_len = data[data_names[0]].shape[0]

        data = torch.cat([torch.tensor(self.data[name]).to(device = self.device, 
                                                           dtype = self.dtype) if \
                          ~isinstance(self.data[name],torch.Tensor) \
                          else self.data[name] \
                          for name in data_names], -1)

        # data = data - data.mean(0, keepdims = True)

        data = data - moving_average(data, torch.hann_window(hann_window_len)) if hann_window_len is not None else data

        data = data.t().unsqueeze(1)

        xcorr_ = torch.nn.functional.conv1d(data, data, padding=(self.record_len - 1))

        self.xcorr = {}
        for i1, name1 in enumerate(data_names):
            for i2, name2 in enumerate(data_names):
                self.xcorr[f"{name2}[t]*{name1}[t-lag]"] = xcorr_[i1, i2].flip(0)[-self.record_len:] / self.record_len

        self.lags = torch.arange(self.record_len) * self.dt

    def plot_ts(self, data_names=None, domain='time', hann_window_len=None, data_units=None, xlim=None,
                figsize=None, title_size=20, xlabel_size=20, ylabel_size=20, fig_num=1):
        """
        Plot the time series data.

        Args:
            data_names (list): Names of the data columns to plot. Default is None (all columns).
            domain (str): Domain of the plot. Can be 'time' or 'frequency'. Default is 'time'.
            hann_window_len (int): Length of the Hanning window for data smoothing. Default is None (no smoothing).
            data_units (list): Units of the data columns. Default is None.
            xlim (tuple): Limits of the x-axis. Default is None.
            figsize (tuple): Size of the figure. Default is None.
            title_size (int): Font size of the title. Default is 20.
            xlabel_size (int): Font size of the x-axis label. Default is 20.
            ylabel_size (int): Font size of the y-axis label. Default is 20.
            fig_num (int): Figure number. Default is 1.
        """
        data, input_size = [], []
        for name in data_names:
            data_ = self.data[name]
            if ~isinstance(data_, torch.Tensor):
                data_ = torch.tensor(data_).to(device = self.device, dtype = self.dtype)
                
            data.append(data_)
            input_size.append(data_.shape[-1])

        data = torch.cat(data, -1)

        data_names = [name for name in data if name != self.time_name] if data_names is None else data_names
        num_data = len(data_names)

        self.record_len = data.shape[0]

        if domain == 'time':
            self.time = self.data[self.time_name]
            xaxis = self.time
            xlabel = f"Time [{self.time_unit}]"
        elif domain == 'frequency':
            data = data - moving_average(data, torch.hann_window(hann_window_len)) if hann_window_len is not None else data
            self.freq, data, _ = fft(data, fs=1 / self.dt)
            xaxis = self.freq
            xlabel = f"Frequency [1/{self.time_unit}]"

        fig, ax = plt.subplots(num_data, 1, figsize=figsize, num=fig_num)

        j = 0
        for i, name in enumerate(data_names):
            ax_i = ax[i] if num_data > 1 else ax
            ax_i.plot(xaxis, data[:, j:(j + input_size[i])], 'k', label=name)
            ax_i.set_title(name, fontsize=title_size)
            ax_i.legend()
            if i == num_data - 1:
                ax_i.set_xlabel(xlabel, fontsize=xlabel_size)
            if data_units is not None:
                ax_i.set_ylabel(data_units[i], fontsize=ylabel_size)
            ax_i.set_xlim(xlim)
            ax_i.grid()
            ax_i.legend(loc='upper right')

            j += input_size[i]

        fig.tight_layout()

    def plot_xcorr(self, domain='time', hann_window_len=None, data_units=None, xlim=None, figsize=None,
                   title_size=20, xlabel_size=20, ylabel_size=20, fig_num=1):
        """
        Plot the cross-correlation between time series data.

        Args:
            domain (str): Domain of the plot. Can be 'time' or 'frequency'. Default is 'time'.
            hann_window_len (int): Length of the Hanning window for data smoothing. Default is None (no smoothing).
            data_units (list): Units of the data columns. Default is None.
            xlim (tuple): Limits of the x-axis. Default is None.
            figsize (tuple): Size of the figure. Default is None.
            title_size (int): Font size of the title. Default is 20.
            xlabel_size (int): Font size of the x-axis label. Default is 20.
            ylabel_size (int): Font size of the y-axis label. Default is 20.
            fig_num (int): Figure number. Default is 1.
        """
        lags = self.lags
        fig, ax = plt.subplots(len(self.xcorr) // 2, len(self.xcorr) // 2, figsize=figsize, num=fig_num)

        xcorr_names = list(self.xcorr)

        i = -1
        for ix in range(len(self.xcorr) // 2):
            for iy in range(len(self.xcorr) // 2):
                ax_ixiy = ax[ix, iy] if len(self.xcorr) // 2 > 1 else ax

                i += 1
                xcorr_ = self.xcorr[xcorr_names[i]].clone()

                if domain == 'frequency':
                    xaxis, xcorr_, _ = fft(xcorr_, fs=1 / self.dt)
                    xlabel = f"Frequency [1/{self.time_unit}]"
                else:
                    xaxis = lags
                    xlabel = f"Lags [{self.time_unit}]"

                ax_ixiy.plot(xaxis, xcorr_)
                ax_ixiy.set_title(xcorr_names[i], fontsize=title_size)
                if ix == len(self.xcorr) // 2 - 1:
                    ax_ixiy.set_xlabel(xlabel, fontsize=xlabel_size)
                ax_ixiy.set_xlim(xlim)
                ax_ixiy.grid()

        fig.tight_layout()
