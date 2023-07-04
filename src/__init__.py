print("Initializing TorchTimeSeries package...")

import importlib, pkgutil

# __all__ = ['FeatureTransform',
#            'Loss', 
#            'fft', 
#            'periodogram', 
#            'moving_average', 
#            'butter', 
#            'fill', 
#            'Interpolator', 
#            'remove_outliers', 
#            'BaselineModel', 
#            'Polynomial', 
#            'LRU', 
#            'HiddenLayer', 
#            'ModulationLayer',
#            'LegendreModulator',
#            'ChebychevModulator', 
#            'FourierModulator', 
#            'SigmoidModulator', 
#            'Attention', 
#            'TransformerEncoderLayer',
#            'TransformerDecoderLayer', 
#            'SequenceModelBase', 
#            'SequenceModel', 
#            'Seq2SeqModel', 
#            'Embedding', 
#            'PositionalEncoding', 
#            'SequenceDataset',                        
#            'SequenceDataloader',
#            'DataModule',
#            'SequenceModule']

# for module_name in __all__:
#     module = importlib.import_module(f'.{module_name}', __name__)
#     globals()[module_name] = module

package = 'src'
__all__ = []

for module_info in pkgutil.iter_modules([package]):
    module_name = module_info.name
    module = importlib.import_module(f'{package}.{module_name}')
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if isinstance(attribute, type):
            globals()[attribute_name] = attribute
            __all__.append(attribute_name)

print("Done")
