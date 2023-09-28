def clear_modules(glbs = globals(), lcls = locals()):
  vars_to_del = []
  for name, var in glbs.items():
    if ('SequenceModule' in name) | ('DataModule' in name):      
      vars_to_del.append(name)
    
  for name in vars_to_del:
    del glbs[name]
    glbs[name] = None
  
  vars_to_del = []
  for name, var in lcls.items():
    if ('SequenceModule' in name) | ('DataModule' in name):
      vars_to_del.append(name)
    
  for name in vars_to_del:
    del lcls[name]
    lcls[name] = None
