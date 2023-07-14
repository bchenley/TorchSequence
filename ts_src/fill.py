def fill(X, steps, interp_kind = 'linear'):
  '''
  Fills missing values in a dataset using interpolation.

  Args:
      X: The input dataset.
      steps: The time steps associated with the data points.
      interp_kind: The kind of interpolation method to use.

  Returns:
      X: The dataset with missing values filled using interpolation.
  '''
  for i in range(X.shape[-1]):
      X_i = X[:, i].copy()

      interpolator = Interpolator(kind=interp_kind)

      if np.any(np.isnan(X_i)):
          X_i_notnan = X_i[~np.isnan(X_i)]
          steps_i_notnan = steps[~np.isnan(X_i)]

          interpolator.fit(steps_i_notnan, X_i_notnan)

          X[:, i] = interpolator.interp_fn(steps)

  return X
