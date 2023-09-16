print("Initializing Diabetes project...")

import importlib

__all__ = ['collect_data',
           'describe']

for module_name in __all__:
    module = importlib.import_module(f'.{module_name}', __name__)
    globals()[module_name] = getattr(module, module_name)

print("Done")
