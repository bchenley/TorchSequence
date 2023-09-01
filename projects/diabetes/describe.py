import pandas as pd
import numpy as np

def describe(data, input_output_names, strata):

  summary = {'id': []}
  summary['Time duration'] = []
  summary['Record length'] = []
  for stratum in strata: summary[stratum] = []

  for name in input_output_names:

    summary[f"mean_{name}"] = []
    summary[f"sdev_{name}"] = []
    summary[f"med_{name}"] = []
    summary[f"min_{name}"] = []
    summary[f"max_{name}"] = []

  for data_ in data:
    time = data_['date']
    summary['id'].append(data_['id'])
    summary['Time duration'].append(time.max() - time.min())
    summary['Record length'].append(len(time))
    for stratum in strata: summary[stratum].append(data_[stratum]) 

    for name in input_output_names:
      if np.sum(data_[name] != 0) > 1:
        if data_[name].shape[-1] == 1:
          mean_ = np.round(data_[name][data_[name] != 0].mean(0),2)
          std_ = np.round(data_[name][data_[name] != 0].std(0),2)
          md_ = np.round(np.median(data_[name][data_[name] != 0], 0),2)
          min_ = np.round(data_[name][data_[name] != 0].min(0),2)
          max_ = np.round(data_[name][data_[name] != 0].max(0),2)
        else:
          mean_ = np.round(data_[name][data_[name] != 0].mean(0),2)
          std_ = np.round(data_[name][data_[name] != 0].std(0),2)
          md_ = np.round(np.median(data_[name][data_[name] != 0], 0),2)
          min_ = np.round(data_[name][data_[name] != 0].min(0),2)
          max_ = np.round(data_[name][data_[name] != 0].max(0),2)
      else:
        mean_ = np.nan
        std_ = np.nan
        md_ = np.nan
        min_ = np.nan
        max_ = np.nan

      summary[f"mean_{name}"].append(mean_)
      summary[f"sdev_{name}"].append(std_)
      summary[f"med_{name}"].append(md_)
      summary[f"min_{name}"].append(min_)
      summary[f"max_{name}"].append(max_)

  summary = pd.DataFrame(summary)

  for name in input_output_names:
    for stat in ['mean', 'min', 'max']:
      value = summary[f"{stat}_{name}"]
      if stat in ['min', 'max']:
        value = np.log(value)
        
      z = (value - value.mean())/value.std()

      low_idx, high_idx = z <= -1.5, z >= 1.5
      mid_idx = ~low_idx & ~high_idx

      summary[f"{stat}_{name}_range"] = np.full((summary.shape[0], 1), np.nan) 
      summary.loc[low_idx, f"{stat}_{name}_range"] = 'low'
      summary.loc[high_idx, f"{stat}_{name}_range"] = 'high'
      summary.loc[mid_idx, f"{stat}_{name}_range"] = 'mid'
  
  return summary
