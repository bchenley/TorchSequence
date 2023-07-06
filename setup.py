from setuptools import setup, find_packages

setup(name = 'TorchTimeSeries',
      version = 0.1,
      description = 'TorchSequence is a PyTorch-based package for time series analysis',
      author = 'Brandon Henley',
      author_email = 'henley.brandon@gmail.com',
      paskages = find_packages(),
      install_requires = [numpy,
                          torch,
                          pytorch_lightning])
