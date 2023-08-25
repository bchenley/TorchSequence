print("Initializing TorchTimeSeries package...")

import importlib
import pkgutil

# __all__ = [name for _,name,_ in pkgutil.iter_modules(['TorchTimeSeries.ts_src'])]

# print(__all__)

__all__ = ['ExploratoryTimeSeriesAnalysis',
           'FeatureTransform',
           'Criterion', 
           'fft', 
           'periodogram', 
           'moving_average', 
           'butter', 
           'fill', 
           'Interpolator', 
           'remove_outliers', 
           'BaselineModel', 
           'Polynomial', 
           'LRU', 
           'HiddenLayer', 
           'ModulationLayer',
           'LegendreModulator',
           'ChebychevModulator', 
           'FourierModulator', 
           'SigmoidModulator', 
           'Attention', 
           'TransformerEncoderLayer',
           'TransformerDecoderLayer', 
           'CNN1D',
           'SequenceModelBase', 
           'SequenceModel', 
           'Seq2SeqModel', 
           'Embedding', 
           'PositionalEncoding', 
           'SequenceDataset',                        
           'SequenceDataloader',
           'TimeSeriesDataModule',
           'SequenceModule',
           'ARIMA',
           'Naive',
           'MovingAverage']

for module_name in __all__:
    module = importlib.import_module(f'.{module_name}', __name__)
    globals()[module_name] = getattr(module, module_name)

           
print("Done")
