import torch
import itertools
import seaborn as sns

class LET():
  def __init__(self,
               num_inputs = 1, num_outputs = 1,
               input_size = [1], output_size = [1],
               lru_num_filterbanks = [1], lru_hidden_size = [1],
               lru_relax_init = [[0.5]], lru_relax_train = [True],
               self_order = [1], cross_order = None,
               fit_type = 'least_squares',
               dt = 1, time_unit = 'S',
               input_units = [None], output_units = [None],
               device = 'cpu', dtype = torch.float32):

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        value = locals_[arg]

        if isinstance(value, list) and any(x in arg for x in ['input_', 'lru_', 'self_']):
          if len(value) == 1:
            setattr(self, arg, value * num_inputs)
          else:
            setattr(self, arg, value)
        elif isinstance(value, list) and any(x in arg for x in ['output_']):
          if len(value) == 1:
            setattr(self, arg, value * num_outputs)
          else:
            setattr(self, arg, value)
        else:
            setattr(self, arg, value)

    self.filterbank = torch.nn.ModuleList([])

    for i in range(self.num_inputs):
      self.filterbank.append(LRU(input_size = self.input_size[i],
                                 hidden_size = self.lru_hidden_size[i],
                                 num_filterbanks = self.lru_num_filterbanks[i],
                                 relax_init = self.lru_relax_init[i],
                                 relax_train = self.lru_relax_train[i],
                                 device = self.device, dtype = self.dtype))

    self.self_kernels, self.cross_kernels = {}, {}

  def get_self_terms(self, fb_output, Q, input_idx):

    num_samples, seq_len, num_filterbanks, hidden_size = fb_output.shape

    self.term_names = self.term_names or []

    all_self_results = []

    # For each order q from 1 to Q
    for q in range(1, Q + 1):
      results = []

      if q == 1:
        # Only products within each filterbank for q = 1
        for f in range(num_filterbanks):
          for h in range(hidden_size):
            temp = fb_output[:, :, f, h]
            results.append(temp.unsqueeze(-1))
            self_term_name = f"i{input_idx+1}_q{q}_f{f+1}_h{h+1}"
            self.term_names.append(self_term_name)

      else:

        # 1. Products within each filterbank:
        for f in range(num_filterbanks):
          for comb in itertools.combinations_with_replacement(range(hidden_size), q):
            temp = torch.ones((num_samples, seq_len), dtype=fb_output.dtype, device=fb_output.device)
            for idx in comb:
                temp *= fb_output[:, :, f, idx]
            results.append(temp.unsqueeze(-1))

            h_str = ''.join([f"h{h+1}" for h in comb])
            self_term_name = f"i{input_idx+1}_q{q}_f{f+1}_{h_str}"
            self.term_names.append(self_term_name)

          # 2. Cross products across filterbanks:
          for f_comb in itertools.combinations(range(num_filterbanks), q):
            for h_comb in itertools.product(range(hidden_size), repeat=q):
              temp = torch.ones((num_samples, seq_len), dtype=fb_output.dtype, device=fb_output.device)
              f_str = ''.join([f"f{f+1}" for f in f_comb])
              h_str = ''.join([f"h{h+1}" for h in h_comb])
              self_term_name = f"i{input_idx+1}_q{q}_{f_str}_{h_str}"
              self.term_names.append(self_term_name)

              for f, h in zip(f_comb, h_comb):
                  temp *= fb_output[:, :, f, h]

              results.append(temp.unsqueeze(-1))

      # Stacking results for each q order
      all_self_results.append(torch.cat(results, -1))

    return all_self_results

  def get_cross_terms(self, self_fb_outputs, Q):

    num_samples, seq_len = self_fb_outputs[0].shape[:2]

    all_cross_results = []

    self.term_names = self.term_names or []

    for q in range(2, Q + 1):
      cross_results_for_q = []

      # Iterate over combinations of filter bank outputs of size q
      for fb_comb in itertools.combinations(self_fb_outputs, q):

        # This list comprehension computes the Cartesian product of filter banks and hidden variables
        # across all tensors in fb_comb
        for combined_idxs in itertools.product(*[[(fb, h) for fb in range(tensor.shape[2]) for h in range(tensor.shape[3])] for tensor in fb_comb]):

          product = torch.ones(num_samples, seq_len,
                               dtype = self_fb_outputs[0].dtype,
                               device = self_fb_outputs[0].device)

          filterbank_hidden_idxs = []
          input_idxs = []
          for tensor_idx, (fb, h) in enumerate(combined_idxs):
            product *= fb_comb[tensor_idx][:, :, fb, h]
            filterbank_hidden_idxs.append((fb, h))
            input_idxs.append(tensor_idx)

          cross_results_for_q.append(product.unsqueeze(-1))

          i_str = f"i{''.join(map(str, input_idxs))}"
          q_str = f"q{q}"
          fb_h_strs = [f"f{fb+1}h{h+1}" for fb,h in filterbank_hidden_idxs]

          cross_term_name = f"{i_str}_{q_str}_{'_'.join(fb_h_strs)}"
          self.term_names.append(cross_term_name)

      # Concatenate the results for this q value and append to the main list
      all_cross_results.append(torch.cat(cross_results_for_q, -1))

    return all_cross_results

  def fit(self,
          input, target,
          hiddens = None):

    self.term_names = ['q0']

    if self.fit_type == 'least_squares':
      with torch.no_grad():
        fb_output, _ = self.generate_filterbank_outputs(input, hiddens)
        self.coefs = torch.pinverse(fb_output) @ target

    self.coef_dict = {}
    for term, coef in zip(self.term_names, self.coefs.squeeze()):
      self.coef_dict[term] = coef.item()

  def generate_filterbank_outputs(self, input, hiddens = None):

    num_samples, input_len, total_input_size = input.shape

    hiddens = hiddens or self.init_hiddens()

    fb_output = [torch.ones(num_samples, input_len, 1).to(device = self.device,
                                                          dtype = self.dtype)]

    self_fb_output = [[] for _ in range(self.num_inputs)]
    self.self_fb_output = []
    for i,input_i in enumerate(input.split(self.input_size, -1)):

      self_fb_output[i], hiddens[i] = self.filterbank[i](input_i, hiddens[i])

      self_fb_output[i] = self.get_self_terms(self_fb_output[i], self.self_order[i], i)

      self.self_fb_output.append(torch.cat(self_fb_output[i], -1))

      fb_output.append(self.self_fb_output[-1])

    if self.cross_order is not None:
      cross_fb_output = self.get_cross_terms([fbo[0].reshape(num_samples, input_len, self.filterbank[i].num_filterbanks, self.filterbank[i].hidden_size) \
                                              for i,fbo in enumerate(self_fb_output)], self.cross_order)

      fb_output.append(torch.cat(cross_fb_output, -1))

    fb_output = torch.cat(fb_output, -1)

    return fb_output, hiddens

  def init_hiddens(self):
    return [None for _ in range(self.num_inputs)]

  def __call__(self, input, hiddens = None):

    fb_output, hiddens = self.generate_filterbank_outputs(input, hiddens)

    output = fb_output @ self.coefs

    return output, hiddens

  def generate_laguerre_functions(self):

    self.basis, self.lags = [], []
    for i in range(self.num_inputs):
      self.basis.append(self.filterbank[i].generate_laguerre_functions())

      lags_i = torch.arange(0, self.basis[i].shape[0]).to(device = self.device,
                                                          dtype = self.dtype) * self.dt

      self.lags.append(lags_i)

  def compute_self_kernels(self):

    if not hasattr(self, 'basis'): self.generate_laguerre_functions()

    for i in range(self.num_inputs):
      hidden_size = self.filterbank[i].hidden_size
      num_filterbanks = self.filterbank[i].num_filterbanks

      second_order_kernel = None

      for q in range(1, np.min([2, self.self_order[i]])+1):
        if q == 1:
          first_order_kernel = torch.zeros(self.basis[i].shape[0]).to(device = self.device,
                                                                      dtype = self.dtype)

          for f in range(num_filterbanks):
            coef_f = []
            for h in range(hidden_size):
              coef_f.append(self.coef_dict[f"i{i+1}_q1_f{f+1}_h{h+1}"])
            coef_f = torch.tensor(coef_f).to(device = self.device,
                                            dtype = self.dtype)
            first_order_kernel += self.basis[i][:, f, :] @ coef_f

        elif q == 2:

          second_order_kernel = torch.zeros((self.basis[i].shape[0], self.basis[i].shape[0])).to(device = self.device,
                                                                                                 dtype = self.dtype)

          for f in range(num_filterbanks):
            for h1 in range(hidden_size):
              for h2 in range(h1, hidden_size):
                coef_f_h1h2 = self.coef_dict[f"i{i+1}_q2_f{f+1}_h{h1+1}h{h2+1}"]
                if h1 != h2:
                  coef_f_h1h2 /= 2

                second_order_kernel += coef_f_h1h2 * self.basis[i][:, f, h1].reshape(-1, 1) @ self.basis[i][:, f, h2].reshape(1, -1)

      self.self_kernels[f"i{i+1}_q1"] = first_order_kernel
      self.self_kernels[f"i{i+1}_q2"] = second_order_kernel

  def compute_cross_kernels(self, input_idxs):

    if not hasattr(self, 'basis'): self.generate_laguerre_functions()

    input_idx_1, input_idx_2 = input_idxs

    hidden_size_1, num_filterbanks_1 = self.filterbank[input_idx_1].hidden_size, self.filterbank[input_idx_1].num_filterbanks
    hidden_size_2, num_filterbanks_2 = self.filterbank[input_idx_2].hidden_size, self.filterbank[input_idx_2].num_filterbanks

    cross_kernel = torch.zeros((self.basis[input_idx_1].shape[0], self.basis[input_idx_2].shape[0])).to(device = self.device,
                                                                                                         dtype = self.dtype)

    for f1 in range(num_filterbanks_1):
      for f2 in range(num_filterbanks_2):
        for h1 in range(hidden_size_1):
          for h2 in range(hidden_size_2):
            coef_f1f2_h1h2 = self.coef_dict[f"i{input_idx_1+1}i{input_idx_2+1}_q2_f{f1+1}f{f2+1}_h{h1+1}h{h2+1}"]

            cross_kernel += coef_f1f2_h1h2 * self.basis[input_idx_1][:, f1, h1].reshape(-1, 1) @ self.basis[input_idx_2][:, f2, h2].reshape(1, -1)

    self.cross_kernels[f"i{input_idx_1+1}i{input_idx_2+1}_q2"] = cross_kernel

  def plot_self_kernels(self, input_idx, figsize = None, fig_num = 1, cmap = cm.coolwarm):

    cols = np.min([2, self.self_order[input_idx]])

    fig = plt.figure(num = fig_num, figsize = figsize or (10*cols, 10))
    ax1 = fig.add_subplot(1, cols, 1)

    lags = self.lags[input_idx]

    ax1.plot(lags, self.self_kernels[f"i{input_idx+1}_q1"].cpu(), 'b')

    ax1.set_title(f"First Order Kernel for Input {input_idx+1}")
    if self.time_unit is not None: ax1.set_xlabel(self.time_unit)
    if self.input_units[input_idx] is not None: ax1.set_ylabel(self.input_units[input_idx])
    ax1.grid()

    if cols > 1:
      ax2 = fig.add_subplot(1, cols, 2, projection = '3d')

      lag_1, lag_2 = np.meshgrid(lags, lags, indexing = 'xy')
      surf = ax2.plot_surface(lag_1, lag_2, self.self_kernels[f"i{input_idx+1}_q2"].cpu(), cmap = cmap)

      ax2.set_title(f"Seconder Order Kernel for Input {input_idx+1}")
      if self.time_unit is not None:
        ax2.set_xlabel(self.time_unit)
        ax2.set_ylabel(self.time_unit)
        if self.input_units[input_idx] is not None: ax2.set_zlabel(fr"${{self.input_units[input_idx]}}^2$")

      fig.colorbar(surf, ax = ax2)
      ax2.grid()

      plt.tight_layout()

  def search_hyperparams(self,
                         input_idx,
                         train_data, test_data = None,
                         criterion = 'bic',
                         self_order_range = [1],
                         num_filterbanks_range = [1],
                         relax_range = torch.arange(0.1, 0.95, 0.05),
                         hidden_size_range = [3, 4, 5, 6, 7]):

    results = {'input': input_idx}

    best_loss = torch.inf

    total_combinations = (len(self_order_range) *
                          len(hidden_size_range) *
                          len(num_filterbanks_range))

    progress_bar = tqdm(total=total_combinations, desc="Initializing search") # , ncols=100)

    train_input, train_target = train_data
    if test_data is not None:
      test_input, test_target = test_data
    else:
      test_input, test_target = train_data

    results[f"train_{criterion}"] = []
    results[f"test_{criterion}"] = []

    with torch.no_grad():
      for self_order in self_order_range:
        self.self_order[input_idx] = self_order
        for hidden_size in hidden_size_range:
          self.filterbank[input_idx].hidden_size = hidden_size

          for num_filterbanks in num_filterbanks_range:
            self.filterbank[input_idx].num_filterbanks = num_filterbanks

            relax_combos = list(itertools.product(relax_range,
                                                  repeat = num_filterbanks))

            train_loss_relax, test_loss_relax = [], []
            for relax in relax_combos:
              self.filterbank[input_idx].relax = torch.nn.Parameter(torch.tensor(relax))
              self.filterbank[input_idx] = self.filterbank[input_idx].to(device = self.device,
                                                                         dtype = self.dtype)

              self.fit(input = train_input, target = train_target)

              train_prediction, _ = self(train_input)
              test_prediction, _ = self(test_input)

              train_loss = Criterion(criterion, (0,1))(train_prediction, train_target, num_params = self.coefs.numel())
              test_loss = Criterion(criterion, (0,1))(test_prediction, test_target, num_params = self.coefs.numel())

              train_loss_relax.append(train_loss)
              test_loss_relax.append(test_loss)

            train_loss_relax = torch.tensor(train_loss_relax).to(train_target)
            test_loss_relax = torch.tensor(test_loss_relax).to(test_target)

            train_loss_min = train_loss_relax.min()
            test_loss_min = test_loss_relax.min()

            results[f"train_{criterion}"].append(train_loss_min)
            results[f"test_{criterion}"].append(test_loss_min)

            best_relax_q_h_f = list(relax_combos[test_loss_min.argmin()])
            best_relax_q_h_f = [torch.round(x, decimals = 2) for x in best_relax_q_h_f]

            if test_loss_min < best_loss:
              results['best_order'] = self_order
              results['best_hidden_size'] = hidden_size
              results['best_num_filterbanks'] = num_filterbanks
              results['best_relax'] = best_relax_q_h_f
              results[f"best_{criterion}"] = test_loss_min

            progress_bar.set_description(f"Best result: {criterion.upper()} = {test_loss_min:.2f}, "
                                         f"Order = {results['best_order']}, "
                                         f"Hidden = {results['best_hidden_size']}, "
                                         f"Filterbanks = {results['best_num_filterbanks']}, "
                                         f"α = {', '.join([f'{val:.2f}' for val in results['best_relax']])}")
            progress_bar.update(1)

    results[f"train_{criterion}"] = torch.stack(results[f"train_{criterion}"])
    results[f"test_{criterion}"] = torch.stack(results[f"test_{criterion}"])

    progress_bar.close()

    return results
  
  def predict(self,
              input, steps,
              hiddens = None,
              start = None, end = None,
              input_output_idx = None,
              output_input_idx = None,
              output_transforms = None):

    num_samples, input_len, input_size = input.shape
    
    start = start or 0
    end = end or input.shape[1]

    with torch.no_grad():

      prediction, hiddens = self(input = input,
                                 hiddens = hiddens)

      while prediction.shape[1] < end:

        input_ = torch.zeros((num_samples, 1, input_size)).to(input)
        if (input_output_idx is not None) & (output_input_idx is not None):
          input_[..., output_input_idx] = prediction[:, -1:, input_output_idx]

        prediction_, hiddens = self(input = input_, hiddens = hiddens)

        prediction = torch.cat((prediction, prediction_), 1)

        steps = torch.cat((steps, steps[:, -1:]+1), 1)

    if output_transforms:
      for sampled_idx in range(num_samples):
        j = 0
        for i in range(self.num_outputs):
          prediction[sampled_idx, :, j:(j+self.output_size[i])] = output_transforms[i].inverse_transform(prediction[sampled_idx, :, j:(j+self.output_size[i])])
          j += self.output_size[i]

    prediction = prediction[:, start:]
    steps = steps[:, start:][:, -prediction.shape[1]:]

    prediction_time = steps * self.dt

    return prediction, prediction_time

  def forecast(self,
               input, steps,
               hiddens = None,
               num_forecast_steps = 1,
               input_output_idx = None,
               output_input_idx = None,
               output_transforms = None):

    forecast, forecast_time = self.predict(input = input, steps = steps,
                                            hiddens = hiddens,
                                            start = input.shape[1],
                                            end = input.shape[1] + num_forecast_steps,
                                            input_output_idx = input_output_idx,
                                            output_input_idx = output_input_idx,
                                            output_transforms = output_transforms)

    return forecast, forecast_time
