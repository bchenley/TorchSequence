import torch
import numpy as np

from src import SequenceModelBase, LRU, HiddenLayer, ModulationLayer

class SequenceModel(torch.nn.Module):
  def __init__(self,
               num_inputs, num_outputs,
               #
               input_size = [1], output_size = [1], seq_len = [None],
               stateful = False,
               dt = 1,
               ## Sequence base parameters
               # type
               base_hidden_size = [1],
               base_type = ['gru'], base_num_layers = [1],
               base_enc2dec_bias = [False],
               encoder_output_size = None,
               # GRU/LSTM parameters
               base_rnn_bias = [True],
               base_rnn_dropout_p = [0],
               base_rnn_bidirectional = [False],
               base_rnn_attn = [False],
               base_encoder_bias = [False], base_decoder_bias = [False],
               base_rnn_weight_reg = [[0.001, 1]], base_rnn_weight_norm = [None],
               # LRU parameters
               base_relax_init = [[0.5]], base_relax_train = [True], base_relax_minmax = [[0.1, 0.9]], base_num_filterbanks = [1],
               # CNN parameters
               base_cnn_kernel_size = [(1,)], base_cnn_stride = [(1,)], base_cnn_padding = [(0,)], base_cnn_dilation = [(1,)], base_cnn_groups = [1], base_cnn_bias = [False],
               # Transformer parameters
               base_seq_type = ['encoder'],
               base_transformer_embedding_type = ['time'], base_transformer_embedding_bias = [False], base_transformer_embedding_activation = ['identity'],
               base_transformer_embedding_weight_reg = [[0.001, 1]], base_transformer_embedding_weight_norm = [2], base_transformer_embedding_dropout_p = [0.0],
               base_transformer_positional_encoding_type = ['absolute'],
               base_transformer_dropout1_p = [0.], base_transformer_dropout2_p = [0.], base_transformer_dropout3_p = [0.],
               base_transformer_linear1_bias = [False], base_transformer_linear2_bias = [False],
               base_transformer_linear1_weight_reg = [[0.001, 1]], base_transformer_linear1_weight_norm = [2],
               base_transformer_linear2_weight_reg = [[0.001, 1]], base_transformer_linear2_weight_norm = [2],
               base_transformer_feedforward_activation = ['relu'],
               base_transformer_feedforward_degree = [2], base_transformer_coef_init = [None], base_transformer_coef_train = [True], base_transformer_coef_reg = [[0.001, 1.]], base_transformer_zero_order = [False],
               base_transformer_scale_self_attn_residual_connection = [False],
               base_transformer_scale_cross_attn_residual_connection = [False],
               base_transformer_scale_feedforward_residual_connection = [False],
               base_transformer_layer_norm = [True],
               # attention parameters
               base_num_heads = [1], base_transformer_dim_feedforward = [2048],
               base_self_attn_type = ['dot'], base_multihead_attn_type = ['dot'],
               base_memory_is_causal = [False], base_tgt_is_causal = [True],
               base_query_dim = [None], base_key_dim = [None], base_value_dim = [None],
               base_query_weight_reg = [[0.001, 1]], base_query_weight_norm = [2], base_query_bias = [False],
               base_key_weight_reg = [[0.001, 1]], base_key_weight_norm = [2], base_key_bias = [False],
               base_value_weight_reg = [[0.001, 1]], base_value_weight_norm = [2], base_value_bias = [False],
               base_gen_weight_reg = [[0.001, 1]], base_gen_weight_norm = [2], base_gen_bias = [False],
               base_concat_weight_reg = [[0.001, 1]], base_concat_weight_norm = [2], base_concat_bias = [False],
               base_attn_dropout_p = [0.], base_average_attn_weights = [False],
               base_constrain = False, base_penalize = False,
               ##
               # hidden layer parameters
               hidden_out_features = [0], hidden_bias = [False], hidden_activation = ['identity'], hidden_degree = [1],
               hidden_coef_init = [None], hidden_coef_train = [True], hidden_coef_reg = [[0.001, 1]], hidden_zero_order = [False],
               hidden_softmax_dim = [-1],
               hidden_constrain = [False], hidden_penalize = [False],
               hidden_dropout_p = [0.],
               # interaction layer
               interaction_out_features = 0, interaction_bias = False, interaction_activation = 'identity',
               interaction_degree = 1, interaction_coef_init = True, interaction_coef_train = True,
               interaction_coef_reg = [0.001, 1], interaction_zero_order = False, interaction_softmax_dim = -1,
               interaction_constrain = False, interaction_penalize = False,
               interaction_dropout_p = 0.,
               # modulation layer
               modulation_window_len = None, modulation_associated = False,
               modulation_legendre_degree = None, modulation_chebychev_degree = None,
               modulation_num_freqs = None, modulation_freq_init = None, modulation_freq_train = True,
               modulation_phase_init = None, modulation_phase_train = True,
               modulation_num_sigmoids = None,
               modulation_slope_init = None, modulation_slope_train = True, modulation_shift_init = None, modulation_shift_train = True,
               modulation_weight_reg = [0.001, 1.0], modulation_weight_norm = 2,
               modulation_zero_order = True,
               modulation_bias = True, modulation_pure = False,
               # output layer
               output_associated = [True],
               output_bias = [True], output_activation = ['identity'], output_degree = [1],
               output_coef_init = [None], output_coef_train = [True], output_coef_reg = [[0.001, 1]], output_zero_order = [False], output_softmax_dim = [-1],
               output_constrain = [False], output_penalize = [False],
               output_dropout_p = [0.],
               #
               device = 'cpu', dtype = torch.float32):

    super(SequenceModel, self).__init__()

    self.to(device = device, dtype = dtype)

    locals_copy = locals().copy() # copy the local variables
    for arg in locals_copy:
      value = locals_copy[arg]
      if isinstance(value, list) and any(x in arg for x in ['seq_type', 'input_size', 'base_', 'decoder_', 'hidden_', 'attn_']):
        if len(value) == 1:
          setattr(self, arg, value * num_inputs)
      elif isinstance(value, list) and any(x in arg for x in ['output_size', 'output_']):
        if len(value) == 1:
          setattr(self, arg, value * num_outputs)

    seq_base, hidden_layer = torch.nn.ModuleList([]), torch.nn.ModuleList([])
    for i in range(num_inputs):
      # input-associated sequence layer
      if base_transformer_feedforward_activation[i] == 'identity':
        base_transformer_dim_feedforward[i] = base_hidden_size[i]

      seq_base_i = SequenceModelBase(input_size = input_size[i],
                                      hidden_size = base_hidden_size[i],
                                      seq_len = seq_len[i],
                                      # type
                                      base_type = base_type[i], num_layers = base_num_layers[i],
                                      encoder_bias = base_encoder_bias[i], decoder_bias = base_decoder_bias[i],
                                      # GRU/LSTM parameters
                                      rnn_bias = base_rnn_bias[i],
                                      rnn_dropout_p = base_rnn_dropout_p[i],
                                      rnn_bidirectional = base_rnn_bidirectional[i],
                                      rnn_attn = base_rnn_attn[i],
                                      rnn_weight_reg = base_rnn_weight_reg[i], rnn_weight_norm = base_rnn_weight_norm[i],
                                      # LRU parameters
                                      relax_init = base_relax_init[i], relax_train = base_relax_train[i], relax_minmax = base_relax_minmax[i], num_filterbanks = base_num_filterbanks[i],
                                      # CNN parameters
                                      cnn_kernel_size = base_cnn_kernel_size[i], cnn_stride = base_cnn_stride[i], cnn_padding = base_cnn_padding[i], cnn_dilation = base_cnn_dilation[i], cnn_groups = base_cnn_groups[i], cnn_bias = base_cnn_bias[i],
                                      # Transformer parameters
                                      encoder_output_size = encoder_output_size, seq_type = base_seq_type[i],
                                      transformer_embedding_type = base_transformer_embedding_type[i], transformer_embedding_bias = base_transformer_embedding_bias[i], transformer_embedding_activation = base_transformer_embedding_activation[i],
                                      transformer_embedding_weight_reg = base_transformer_embedding_weight_reg[i], transformer_embedding_weight_norm = base_transformer_embedding_weight_norm[i], transformer_embedding_dropout_p = base_transformer_embedding_dropout_p[i],
                                      transformer_positional_encoding_type = base_transformer_positional_encoding_type[i],
                                      transformer_dropout1_p = base_transformer_dropout1_p[i], transformer_dropout2_p = base_transformer_dropout2_p[i], transformer_dropout3_p = base_transformer_dropout3_p[i],
                                      transformer_linear1_bias = base_transformer_linear1_bias[i], transformer_linear2_bias = base_transformer_linear2_bias[i],
                                      transformer_linear1_weight_reg = base_transformer_linear1_weight_reg[i], transformer_linear1_weight_norm = base_transformer_linear1_weight_norm[i],
                                      transformer_linear2_weight_reg = base_transformer_linear2_weight_reg[i], transformer_linear2_weight_norm = base_transformer_linear2_weight_norm[i],
                                      transformer_feedforward_activation = base_transformer_feedforward_activation[i],
                                      transformer_feedforward_degree = base_transformer_feedforward_degree[i], transformer_coef_init = base_transformer_coef_init[i], transformer_coef_train = base_transformer_coef_train[i], transformer_coef_reg = base_transformer_coef_reg[i], transformer_zero_order = base_transformer_zero_order[i],
                                      transformer_scale_self_attn_residual_connection = base_transformer_scale_self_attn_residual_connection[i],
                                      transformer_scale_cross_attn_residual_connection = base_transformer_scale_cross_attn_residual_connection[i],
                                      transformer_scale_feedforward_residual_connection = base_transformer_scale_feedforward_residual_connection[i],
                                      transformer_layer_norm = base_transformer_layer_norm[i],
                                      # attention parameters
                                      num_heads = base_num_heads[i], transformer_dim_feedforward = base_transformer_dim_feedforward[i],
                                      self_attn_type = base_self_attn_type[i], multihead_attn_type = base_multihead_attn_type[i],
                                      memory_is_causal = base_memory_is_causal[i], tgt_is_causal = base_tgt_is_causal[i],
                                      query_dim = base_query_dim[i], key_dim = base_key_dim[i], value_dim = base_value_dim[i],
                                      query_weight_reg = base_query_weight_reg[i], query_weight_norm = base_query_weight_norm[i], query_bias = base_query_bias[i],
                                      key_weight_reg = base_key_weight_reg[i], key_weight_norm = base_key_weight_norm[i], key_bias = base_key_bias[i],
                                      value_weight_reg = base_value_weight_reg[i], value_weight_norm = base_value_weight_norm[i], value_bias = base_value_bias[i],
                                      gen_weight_reg = base_gen_weight_reg[i], gen_weight_norm = base_gen_weight_norm[i], gen_bias = base_gen_bias[i],
                                      concat_weight_reg = base_concat_weight_reg[i], concat_weight_norm = base_concat_weight_norm[i], concat_bias = base_concat_bias[i],
                                      attn_dropout_p = base_attn_dropout_p[i],
                                      average_attn_weights = base_average_attn_weights[i],
                                      # always batch first
                                      batch_first = True,
                                      #
                                      device = device, dtype = dtype)

      seq_base.append(seq_base_i)
      #

      # input-associated hidden layer
      if hidden_out_features[i] > 0:
        if base_hidden_size[i] > 0:
          if base_type[i] == 'lru':
            hidden_in_features_i = base_hidden_size[i]*len(base_relax_init[i])
          elif base_type[i] in ['gru', 'lstm']:
            hidden_in_features_i = base_hidden_size[i]*(1+base_rnn_bidirectional[i])
          else:
            hidden_in_features_i = base_hidden_size[i]
        else:
          input_size = input_size[i]

        hidden_layer_i = HiddenLayer(# linear transformation
                                     in_features = hidden_in_features_i, out_features = hidden_out_features[i],
                                     bias = hidden_bias[i],
                                     # activation
                                     activation = hidden_activation[i],
                                     # polynomial parameters
                                     degree = hidden_degree[i],
                                     coef_init = hidden_coef_init[i], coef_train = hidden_coef_train[i], coef_reg = hidden_coef_reg[i],
                                     zero_order = hidden_zero_order[i],
                                     # softmax parameter
                                     softmax_dim = hidden_softmax_dim[i],
                                     dropout_p = hidden_dropout_p[i],
                                     device = device, dtype = dtype)
      else:
        hidden_layer_i = torch.nn.Identity()

      hidden_layer.append(hidden_layer_i)
      #

    # interaction layer
    if interaction_out_features > 0:
      if np.sum(hidden_out_features) > 0:
        interaction_in_features = int(np.sum(hidden_out_features))
      else:
        interaction_in_features = 0
        for i in range(num_inputs):
          interaction_in_features += base_hidden_size[i]*(1+base_rnn_bidirectional[i])
        else:
          interaction_in_features += base_transformer_dim_feedforward[i]

      interaction_layer = HiddenLayer(# linear transformation
                                      in_features = interaction_in_features, out_features = interaction_out_features,
                                      bias = interaction_bias,
                                      # activation
                                      activation = interaction_activation,
                                      # polynomial parameters
                                      degree = interaction_degree,
                                      coef_init = interaction_coef_init, coef_train = interaction_coef_train, coef_reg = interaction_coef_reg,
                                      zero_order = interaction_zero_order,
                                      # softmax parameter
                                      softmax_dim = interaction_softmax_dim,
                                      dropout_p = interaction_dropout_p,
                                      device = device, dtype = dtype)
    else:
      interaction_layer = torch.nn.Identity()

    # modulation layer
    modulation_layer = None
    if modulation_window_len is not None:
      if interaction_out_features > 0:
        modulation_in_features = interaction_out_features
      elif np.sum(hidden_out_features) > 0:
        modulation_in_features = np.sum(hidden_out_features)
      else:
        modulation_in_features = 0
        for i in range(num_inputs):
          modulation_in_features += base_hidden_size[i]*(1+base_rnn_bidirectional[i])
        else:
          modulation_in_features += base_transformer_dim_feedforward[i]
      #

      modulation_layer = ModulationLayer(window_len = modulation_window_len,
                                        in_features = modulation_in_features,
                                        associated = modulation_associated,
                                        legendre_degree = modulation_legendre_degree,
                                        chebychev_degree = modulation_chebychev_degree,
                                        dt = dt,
                                        num_freqs = modulation_num_freqs, freq_init = modulation_freq_init,  freq_train = modulation_freq_init,
                                        phase_init = modulation_phase_init, phase_train = modulation_phase_train,
                                        num_sigmoids = modulation_num_sigmoids,
                                        slope_init = modulation_slope_init, slope_train = modulation_slope_train,
                                        shift_init = modulation_shift_init, shift_train=  modulation_shift_init,
                                        weight_reg = modulation_weight_reg, weight_norm = modulation_weight_norm,
                                        zero_order = modulation_zero_order,
                                        bias = modulation_bias, pure = modulation_pure,
                                        device = device, dtype = dtype)
    #

    # output layer
    output_layer = torch.nn.ModuleList([])
    for i in range(num_outputs):
      if modulation_layer is not None:
        output_input_size_i = modulation_layer.num_modulators
      elif interaction_out_features > 0:
        output_input_size_i = interaction_out_features
      elif np.sum(hidden_out_features) > 0:
        if output_associated[i]:
          output_input_size_i = hidden_out_features[i]
        else:
          output_input_size_i = int(np.sum(hidden_out_features))
      else:
        if output_associated[i]:
          if base_type[i] in ['gru', 'lstm', 'lru']:
            output_input_size_i = base_hidden_size[i]*(1+base_rnn_bidirectional[i])
          elif base_type[i] == 'transformer':
            output_input_size_i = base_hidden_size[i]
          else:
            output_input_size_i = input_size[i]
        else:
          output_input_size_i = 0
          for i in range(num_inputs):
            if base_type[i] in ['gru', 'lstm', 'lru']:
              output_input_size_i += int(base_hidden_size[i]*(1+base_rnn_bidirectional[i]))
            elif base_type[i] == 'transformer':
              output_input_size_i += base_hidden_size[i]
            else:
              output_input_size_i += input_size[i]

      if output_size[i] > 0:
        output_layer_i = HiddenLayer(# linear transformation
                                     in_features = output_input_size_i, out_features = output_size[i],
                                     bias = output_bias[i],
                                     # activation
                                     activation = output_activation[i],
                                     # polynomial parameters
                                     degree = output_degree[i],
                                     coef_init = output_coef_init[i], coef_train = output_coef_train[i], coef_reg = output_coef_reg[i],
                                     zero_order = output_zero_order[i],
                                     # softmax parameter
                                     softmax_dim = output_softmax_dim[i],
                                     dropout_p = output_dropout_p[i],
                                     device = device, dtype = dtype)
      else:
        output_layer_i = torch.nn.Identity()
        if np.sum(hidden_out_features) > 0:
          output_size[i] = hidden_out_features[i]
        elif interaction_out_features > 0:
          output_size[i] = interaction_out_features
        elif output_associated[i]:
          output_size[i] = base_hidden_size[i]*(1 + base_rnn_bidirectional[i])
        else:
          output_size[i] = int(np.sum(np.array(base_hidden_size)*(1+np.array(base_rnn_bidirectional))))

      output_layer.append(output_layer_i)

    self.num_inputs,self.num_outputs = num_inputs, num_outputs
    self.input_size, self.output_size = input_size, output_size

    self.stateful = stateful
    self.seq_base, self.base_hidden_size, self.base_type = seq_base, base_hidden_size, base_type
    self.base_constrain, self.base_penalize = base_constrain, base_penalize

    self.hidden_layer, self.hidden_out_features = hidden_layer, hidden_out_features
    self.hidden_constrain, self.hidden_penalize = hidden_constrain, hidden_penalize

    self.interaction_layer, self.interaction_out_features = interaction_layer, interaction_out_features
    self.interaction_constrain, self.interaction_penalize = interaction_constrain, interaction_penalize

    self.modulation_layer = modulation_layer

    self.output_layer = output_layer
    self.output_associated = output_associated
    self.output_constrain, self.output_penalize = output_constrain, output_penalize

    self.device, self.dtype = device, dtype

  def __repr__(self):
    total_num_params = 0
    total_num_trainable_params = 0
    lines = []
    for name, param in self.named_parameters():
      trainable = 'Trainable' if param.requires_grad else 'Untrainable'
      lines.append(f"{name}: shape = {param.shape}. {param.numel()} parameters. {trainable}")
      total_num_params += param.numel()
      if param.requires_grad: total_num_trainable_params += param.numel()

    lines.append("-------------------------------------")
    lines.append(f"{total_num_params} total parameters.")
    lines.append(f"{total_num_trainable_params} total trainable parameters.")

    return '\n'.join(lines)

  def init_hiddens(self):
    return [None for _ in range(self.num_inputs)]

  def process(self,
              input, hiddens,
              steps = None,
              encoder_output = None):

    # Get the dimensions of the input
    num_samples, input_len, input_size = input.shape

    # List to store the output of hidden layers
    hidden_output = []

    # Process each input in the batch individually
    for i,input_i in enumerate(input.split(self.input_size, -1)):

      # Generate output and hiddens of sequence base for the ith input
      base_output_i, hiddens[i] = self.seq_base[i](input = input_i[:, -1:] \
                                                   if (self.seq_base[i].base_type in ['gru','lstm','lru']) & (self.seq_base[i].seq_type == 'decoder') \
                                                   else input_i,
                                                   hiddens = hiddens[i],
                                                   encoder_output = encoder_output)

      base_output_i = torch.nn.functional.pad(base_output_i,
                                              (0, 0, np.max([0, base_output_i.shape[1]-input_len]), 0),
                                              "constant", 0)

      # Generate hidden layer outputs for ith input, append result to previous hidden layer output of previous inputs
      hidden_output_i = self.hidden_layer[i](base_output_i)
      hidden_output.append(hidden_output_i)

    output_ = torch.cat(hidden_output,-1)

    output_ = self.interaction_layer(output_)

    if self.modulation_layer is not None:
      output_ = self.modulation_layer(output_, steps)

    # For each output
    output = []
    for i in range(self.num_outputs):
      # If ith output layer is "associated" (linked to a single input)
      if self.output_associated[i]:
        # Set the output of the ith hidden layer as input to the ith output layer
        output_input_i = hidden_output[i]
      # Otherwise, pass the entire output of previous layer as the input to the ith output layer
      else:
        output_input_i = output_

      # Generate output of ith output layer, append result to previous outputs
      output_i = self.output_layer[i](output_input_i)
      output.append(output_i)

    # Concatenate outputs into single tensor
    output = torch.cat(output, -1)

    # Apply modulation layer
    if self.modulation_layer is not None:
      output = self.modulation_layer(output, steps)

    return output, hiddens

  def forward(self,
              input, steps = None,
              hiddens = None,
              target = None,
              output_window_idx = None,
              input_mask = None, output_mask = None,
              output_input_idx = None, input_output_idx = None,
              encoder_output= None):

    # Convert inputs to the correct device
    input = input.to(device = self.device)
    steps = steps.to(device = self.device) if steps is not None else None
    output_mask = output_mask.to(device =  self.device) if output_mask is not None else None

    # Get the dimensions of the input
    num_samples, input_len, input_size = input.shape

    # Get total number of steps
    if steps is not None:
      _, num_steps = steps.shape

    # Get the maximum output sequence length
    max_output_len = np.max([len(idx) for idx in output_window_idx]) if output_window_idx is not None else input_len

    # Get the total output size
    total_output_size = np.sum(self.output_size)

    # Initiate hiddens if None or not stateful
    if (hiddens is None) | (not self.stateful) & any(type_ in ['gru', 'lstm', 'lru'] for type_ in self.base_type):
      hiddens = hiddens or self.init_hiddens()

    # Process output and updated hiddens
    if 'encoder' in [base.seq_type for base in self.seq_base]: # model is an encoder
      output, hiddens = self.process(input = input,
                                     steps = steps,
                                     hiddens = hiddens,
                                     encoder_output = encoder_output)
    else: # model is a decoder

      # Prepare input for the next step
      input_, output = input, []
      for n in range(max_output_len):
        output_, hiddens = self.process(input = input_.clone()[:, :(n+1)],
                                        steps = steps[:, :(n+1)] if steps is not None else None,
                                        hiddens = hiddens,
                                        encoder_output = encoder_output)

        output.append(output_[:, -1:])

        if (len(output_input_idx) > 0) & (n < (max_output_len-1)):
          input_[:, (n+1):(n+2), output_input_idx] = target[:, n:(n+1), input_output_idx] if target is not None else output[-1][..., input_output_idx]

      output = torch.cat(output, 1)

    # Only keep the outputs for the maximum output sequence length
    output = output[:, -max_output_len:]

    # Apply the output mask if specified
    if output_mask is not None: output = output*output_mask

    return output, hiddens

  def constrain(self):

    for i in range(self.num_inputs):
      if self.base_constrain[i]:
        self.seq_base[i].constrain()

      if self.hidden_constrain[i]:
         self.hidden_layer[i].constrain()

    if self.interaction_constrain:
       self.interaction_layer.constrain()

    for i in range(self.num_outputs):
      if self.output_constrain[i]:
         self.output_layer[i].constrain()

  def penalize(self):
    loss = 0

    for i in range(self.num_inputs):
      if self.base_penalize[i]:
        loss += self.seq_base[i].penalize()

    if self.hidden_penalize[i]:
      loss += self.hidden_layer[i].penalize()

    if self.interaction_penalize:
      loss += self.interaction_layer.penalize()

    for i in range(self.num_outputs):
      if self.output_penalize[i]:
        loss += self.output_layer[i].penalize()

    return loss

  def generate_impulse_response(self, seq_len):
    with torch.no_grad():
      impulse_response = [None for _ in range(self.num_inputs)]
      for i in range(self.num_inputs):
        # if self.base_type[i] in ['gru', 'lstm', 'lru']:
        impulse_response[i] = [None for _ in range(self.input_size[i])]
        for f in range(self.input_size[i]):
          impulse_i = torch.zeros((1, seq_len, self.input_size[i])).to(device = self.device,
                                                                                  dtype = self.dtype)
          impulse_i[0, 0, f] = 1.

          base_output_if, _ = self.seq_base[i](input = impulse_i)

          weight = self.hidden_layer[i].F[0].weight

          impulse_response[i][f] = base_output_if.squeeze(0) @ weight.t()

    return impulse_response
