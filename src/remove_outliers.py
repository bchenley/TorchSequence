import numpy as np

def remove_outliers(X, steps, abs_max_change=[np.inf], z_change_critical=[7], interp_type='linear'):
  '''
  Remove outliers from the input data.

  Args:
      X: The input data array.
      steps: The corresponding steps array.
      abs_max_change: The absolute maximum change threshold for outlier removal.
      z_change_critical: The z-change critical value for outlier removal.
      interp_type: The type of interpolation to use for filling the gaps left by removed outliers.

  Returns:
      Y: The input data with outliers removed and gaps filled.
      steps: The corresponding steps array after outlier removal.

  '''
  if len(abs_max_change) == 1:
      abs_max_change = abs_max_change * X.shape[-1]
  if len(z_change_critical) == 1:
      z_change_critical = z_change_critical * X.shape[-1]

  x, i_x, interpolator = [], [], []

  for i in range(X.shape[-1]):
      X_i = X[:, i]

      i_all = np.arange(X_i.shape[0], dtype=np.compat.long)
      i_x.append(i_all)

      diff = np.diff(X_i)
      z_diff = (diff - diff.mean()) / diff.std()

      i_discard = np.where((np.abs(diff) > abs_max_change[i]) | (np.abs(z_diff) > z_change_critical[i]))[0]

      if len(i_discard) > 0:
          interpolator.append(Interpolator(kind=interp_type))
      else:
          interpolator.append(None)

      while len(i_discard) > 0:
          j_discard = i_discard[0] + [0, 1]
          j_discard = j_discard[j_discard < len(X_i)]

          j_discard = j_discard[np.abs(X_i.mean() - X_i[j_discard]).argmax()]
          X_i = np.delete(X_i, j_discard)
          i_x[-1] = np.delete(i_x[-1], j_discard)

          diff = np.diff(X_i)
          z_diff = (diff - diff.mean()) / diff.std()

          i_discard = np.where((np.abs(diff) > abs_max_change[i]) | (np.abs(z_diff) > z_change_critical[i]))[0]

      x.append(X_i.reshape(-1, 1))

      print(f"{i+1}/{X.shape[-1]}")

  i_min = np.max([np.min(i) for i in i_x])
  i_max = np.min([np.max(i) for i in i_x])

  i_all = np.arange(i_min, i_max + 1, dtype=np.compat.long)
  steps = steps[i_all]

  for i, interp in enumerate(interpolator):
      if interp is not None:
          interp.fit(i_x[i], x[i])
          x[i] = interp.interp_fn(i_all)

  Y = np.concatenate(x, -1)

  return Y, steps
