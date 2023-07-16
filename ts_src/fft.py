import torch
import pandas as pd
import numpy as np

def fft(x, fs = 1, dim = 0, nfft = None, norm = 'backward',
        device = None, dtype = torch.complex64):
  '''
  Computes the Fast Fourier Transform (FFT) of the input signal.

  Args:
      x: The input signal. If not a torch.Tensor, it will be converted to one.
      fs: The sampling frequency of the input signal.
      dim: The dimension(s) along which to compute the FFT.
      nfft: The number of FFT points. If None, it is set to the size of the input signal along the specified dimension.
      norm: The normalization mode. Options are 'backward' (default) and 'forward'.
      device: The device to perform the computation on. If None, the default device is used.
      dtype: The data type of the output.

  Returns:
      freq: The frequency values corresponding to the FFT.
      x_fft_mag: The magnitude of the FFT coefficients.
      x_fft_phase: The phase of the FFT coefficients.
  '''
  if not isinstance(x, torch.Tensor):
      if isinstance(x, pd.core.frame.DataFrame):
          x = x.values
      x = torch.tensor(x).to(device = device, dtype = dtype)

  # if nfft is None:
  #     nfft = x.shape[dim]
  #     print(f'nfft set to {nfft}')

  s, dim = [nfft, dim if isinstance(dim, int) else (-2, -1)]

  s += np.mod(s, 2)
  x_fft = torch.fft.fftn(x, s = s, dim = dim, norm = norm).to(device = device, 
                                                              dtype = dtype)

  N = int(s // 2)

  if isinstance(dim, int):
      freq = torch.fft.fftfreq(s, d = 1 / fs).to(device = device)[:N]

      x_fft = x_fft.split(N, dim = dim)[0]

      x_fft_mag = 2.0 / s * torch.abs(x_fft)

      x_fft_phase = torch.angle(x_fft)

  elif dim == (-2, -1):
      freq = torch.meshgrid(freq, freq, indexing='ij')

      x_fft_mag = 2 / s * torch.abs(x_fft[..., :N, :N])

      x_fft_phase = torch.angle(x_fft)[..., :N, :N]

  else:
      raise ValueError(f'dim ({dim}) must be 1 or (-2, -1)... for now.')

  return freq, x_fft_mag, x_fft_phase
