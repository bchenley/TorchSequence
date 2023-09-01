import pandas as pd
import numpy as np
import math
import os
import re

def collect_data(t1_path, t1_summary_path,
                 t2_path, t2_summary_path):

  t1_files = [file for file in os.listdir(t1_path) if '.xls' in file]
  t2_files = [file for file in os.listdir(t2_path) if '.xls' in file]

  t1_summary = pd.read_excel(t1_summary_path)
  t2_summary = pd.read_excel(t2_summary_path)

  t1_patient_number = t1_summary['Patient Number'].values
  t1_summary = t1_summary.drop('Patient Number', axis = 1)

  t2_patient_number = t2_summary['Patient Number'].values
  t2_summary = t2_summary.drop('Patient Number', axis = 1)
  
  num_t1_files, num_t2_files = len(t1_files), len(t2_files)
  
  data = []
  for f, file in enumerate(t1_files + t2_files):
    data.append({})
    
    if f < num_t1_files:
      path = t1_path
      patient_number = t1_patient_number
      summary = t1_summary
      data[f]['diabetes'] = '1'
    else:
      path = t2_path
      patient_number = t2_patient_number
      summary = t2_summary
      data[f]['diabetes'] = '2'
    
    data_f = dict(pd.read_excel(os.path.join(path, file))).copy()

    data[f]['id'] = file.split('.')[0]

    data[f]['date'] = data_f['Date'].copy().astype('datetime64[ns]')
    
    data[f]['iu'] = np.full((data[f]['date'].shape[0], 1), 0.)
    data[f]['iu_h'] = np.full((data[f]['date'].shape[0], 1), 0.)
    data[f]['di'] = np.full((data[f]['date'].shape[0], 1), np.nan)
    data[f]['di_g'] = np.full((data[f]['date'].shape[0], 1), 0.)
    for key in data_f.keys():    
      # Glucose
      if 'cgm' in key.lower():
        data[f]['cgm'] = data_f[key].copy().values.reshape(-1, 1)      
        for n in range(data[f]['cgm'].shape[0]):
          if isinstance(data[f]['cgm'][n,0], str):            
            data[f]['cgm'][n,0] = 0.
          elif math.isnan(data[f]['cgm'][n,0]):
            data[f]['cgm'][n,0] = 0.              
        data[f]['cgm'] = data[f]['cgm'].astype('float')
        
      elif 'cbg' in key.lower():
        data[f]['cbg'] = data_f[key].copy().values.reshape(-1, 1)      
        for value in data[f]['cbg']:
          if isinstance(data[f]['cbg'][n,0], str):
            data[f]['cbg'][n,0] = 0.
          elif math.isnan(data[f]['cbg'][n,0]):
            data[f]['cbg'][n,0] = 0.
        data[f]['cbg'] = data[f]['cbg'].astype('float')
      
      # Keton
      if 'keton' in key.lower():
        data[f]['kb'] = data_f[key].copy().values.reshape(-1, 1)      
        for n in range(data[f]['kb'].shape[0]):
          if isinstance(data[f]['kb'][n,0], str):
            data[f]['kb'][n,0] = 0.
          elif math.isnan(data[f]['kb'][n,0]):
            data[f]['kb'][n,0] = 0.          
        data[f]['kb'] = data[f]['kb'].astype('float')
      
      # Insulin
      if ('novolin' in key.lower()) & ('h' not in key.lower()):
        data[f]['iu'] = data_f[key].copy().values.reshape(-1, 1)     
        for n in range(data[f]['iu'].shape[0]):
          if isinstance(data[f]['iu'][n,0], str):
            data[f]['iu'][n,0] = 0.
          elif math.isnan(data[f]['iu'][n,0]):
            data[f]['iu'][n,0] = 0.
        data[f]['iu'] = data[f]['iu'].astype('float')

      # Insulin per hour
      if ('novolin' in key.lower()) & ('h' in key.lower()):
        data[f]['iu_h'] = data_f[key].copy().values.reshape(-1, 1)      
        for n in range(data[f]['iu_h'].shape[0]):
          if isinstance(data[f]['iu_h'][n,0], str):            
            data[f]['iu_h'][n,0] = 0.
          elif math.isnan(data[f]['iu_h'][n,0]):
            data[f]['iu_h'][n,0] = 0.
        data[f]['iu_h'] = data[f]['iu_h'].astype('float')

      # Dietary intake
      if ('dietary intake' in key.lower()):
        data[f]['di'] = data_f[key].copy().values.reshape(-1, 1)
        for n in range(data[f]['di'].shape[0]):
          g = 0.
          if isinstance(data[f]['di'][n,0], str):    
            di_split = data[f]['di'][n,0].split(' ')
            for i, string in enumerate(di_split):
              if string == 'g':
                number_str = re.findall(r"\d+", di_split[i-1])
                g += sum([float(s) for s in number_str])
              # try:                    
              #   g += float(string)
              # except ValueError: 
              #   pass   
          data[f]['di_g'][n,0] = g

      # Dietary intak rate (g/day)
      total_days = (data[f]['date'].max() - data[f]['date'].min()).total_seconds()/3600/24
      data[f]['di_g_r'] = np.sum(data[f]['di_g']) / total_days

    summary_idx = np.where(data[f]['id'] == patient_number)[0].item()
    for label, value in summary.iloc[summary_idx, :].items():
      data[f][label] = value 

    print(f"File {file} complete {f+1}/{num_t1_files + num_t2_files}") 
    
  return data
