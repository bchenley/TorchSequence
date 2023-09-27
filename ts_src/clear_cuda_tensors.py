def clear_cuda_tensors(glbs = globals()):
  vars_to_del = []
  for name, var in glbs.items():
    if isinstance(var, torch.Tensor):
       vars_to_del.append(name)
    elif isinstance(var, list):
      if all(isinstance(var_, torch.Tensor) for var_ in var):
        vars_to_del.append(name)
