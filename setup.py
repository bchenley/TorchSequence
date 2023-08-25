from setuptools import setup, find_packages

setup(name = 'torchts',
      version = '0.1',
      packages = find_packages(), 
      install_requires = ['torch',
                          'pytorch_lightning',
                          'numpy',
                          'pandas',
                          'scikit-learn',
                          'seaborn',
                          'statsmodels',
                          'datetime',
                          'matplotlib',
                          'google',
                          'os'],
      author = "B.C. Henley",
      author_email = 'henley.brandon@gmail.com',
      description = "A package for time series analysis using PyTorch",
      url = 'http://github.com/bchenley/TorchTimeSeries',
      classifiers = ["Programming Language :: Python :: 3",
                     "License :: OSI Approved :: MIT License"]
)
