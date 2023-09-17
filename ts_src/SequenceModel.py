import torch
import numpy as np

from ts_src.SequenceModelBase import SequenceModelBase
from ts_src.LRU import LRU
from ts_src.HiddenLayer import HiddenLayer
from ts_src.ModulationLayer import ModulationLayer

class SequenceModel(torch.nn.Module):
  def __init__(self,
               num_inputs, num_outputs,
               #
               input_size = [1], input_len = [1],
               output_size = [1], output_len = [1],
               base_stateful = [False], process_by_step = False, joint_prediction = False,
               dt = 1,
               norm_type = None, affine_norm = False,               
               store_layer_outputs = False,
               encoder_output_size = None,
               ## Sequence base parameters
               # type
               base_hidden_size = [1],
               base_type = ['gru'], base_num_layers = [1],
               base_enc2dec_bias = [False],
               base_use_last_step = [False],
               # GRU/LSTM parameters
               base_rnn_bias = [True],
               base_rnn_dropout_p = [0],
               base_rnn_bidirectional = [False],
               base_rnn_attn = [False],
               base_encoder_bias = [False], base_decoder_bias = [False],
               base_rnn_weight_reg = [[0.001, 1]], base_rnn_weight_norm = [None],
               # LRU parameters
               base_relax_init = [[0.5]], base_relax_train = [True], base_relax_minmax = [[[0.1, 0.9]]], base_num_filterbanks = [1],
               # CNN parameters
               base_cnn_out_channels = [[1]],
               base_cnn_kernel_size = [[(1,)]], base_cnn_kernel_stride = [[(1,)]],
               base_cnn_padding = [[(0,)]], base_cnn_dilation = [[(1,)]],
               base_cnn_groups = [[1]], base_cnn_bias = [[True]],
               base_cnn_activation = [['identity']],
               base_cnn_degree = [[2]], base_cnn_coef_init = [[None]], base_cnn_coef_train = [[True]], base_cnn_coef_reg = [[[0.001, 1]]], base_cnn_zero_order = [[False]],
               base_cnn_pool_type = [[None]], base_cnn_pool_size = [[(2,)]], base_cnn_pool_stride = [[(1,)]],
               base_cnn_batch_norm = [False], base_cnn_batch_norm_learn = [False],
               base_cnn_causal_pad = [False], base_cnn_dropout_p = [[0.]],
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
               base_constrain = [False], base_penalize = [False],
               ##
               # hidden layer parameters
               hidden_out_features = [0], hidden_bias = [False], hidden_activation = ['identity'], hidden_degree = [1],
               hidden_coef_init = [None], hidden_coef_train = [True], hidden_coef_reg = [[0.001, 1]], hidden_zero_order = [False],
               hidden_softmax_dim = [-1],
               hidden_constrain = [False], hidden_penalize = [False],
               hidden_dropout_p = [0.],
               # interaction layer
               interaction_out_features = 0, interaction_bias = False, interaction_activation = 'identity',
               interaction_degree = 1, interaction_coef_init = None, interaction_coef_train = True,
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
               output_associated = [False],
               output_bias = [True], output_activation = ['identity'], output_degree = [1],
               output_coef_init = [None], output_coef_train = [True], output_coef_reg = [[0.001, 1]], output_zero_order = [False], output_softmax_dim = [-1],
               output_constrain = [False], output_penalize = [False],
               output_dropout_p = [0.],
               output_layer_w_to_1 = [False],
               output_flatten = [None],
               #
               device = 'cpu', dtype = torch.float32):

    super(SequenceModel, self).__init__()

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        value = locals_[arg]

        if isinstance(value, list) and any(x in arg for x in ['input_size', 'input_len','base_', 'decoder_', 'hidden_', 'attn_']):
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

    self.max_input_len, self.max_output_len = np.max(self.input_len), np.max(self.output_len)

    output_size_original = self.output_size.copy()
    
    self.seq_base, self.hidden_layer = torch.nn.ModuleList([]), torch.nn.ModuleList([])
    for i in range(self.num_inputs):

      if self.base_transformer_feedforward_activation[i] == 'identity':
        self.base_transformer_dim_feedforward[i] = self.base_hidden_size[i]

      seq_base_i = SequenceModelBase(input_size = self.input_size[i],
                                     hidden_size = self.base_hidden_size[i],
                                     input_len = self.input_len[i],
                                     use_last_step = self.base_use_last_step[i],
                                     # type
                                     base_type = self.base_type[i], num_layers = self.base_num_layers[i],
                                     encoder_bias = self.base_encoder_bias[i], decoder_bias = self.base_decoder_bias[i],
                                     # GRU/LSTM parameters
                                     rnn_bias = self.base_rnn_bias[i],
                                     rnn_dropout_p = self.base_rnn_dropout_p[i],
                                     rnn_bidirectional = self.base_rnn_bidirectional[i],
                                     rnn_attn = self.base_rnn_attn[i],
                                     rnn_weight_reg = self.base_rnn_weight_reg[i], rnn_weight_norm = self.base_rnn_weight_norm[i],
                                     # LRU parameters
                                     relax_init = self.base_relax_init[i], relax_train = self.base_relax_train[i], relax_minmax = self.base_relax_minmax[i], num_filterbanks = self.base_num_filterbanks[i],
                                     # CNN parameters
                                     cnn_out_channels = self.base_cnn_out_channels[i],
                                     cnn_causal_pad = self.base_cnn_causal_pad[i],
                                     cnn_kernel_size = self.base_cnn_kernel_size[i], cnn_kernel_stride = self.base_cnn_kernel_stride[i], cnn_padding = self.base_cnn_padding[i], cnn_dilation = self.base_cnn_dilation[i], cnn_groups = self.base_cnn_groups[i], cnn_bias = self.base_cnn_bias[i],
                                     cnn_activation = self.base_cnn_activation[i],
                                     cnn_degree = self.base_cnn_degree[i], cnn_coef_init = self.base_cnn_coef_init[i], cnn_coef_train = self.base_cnn_coef_train[i], cnn_coef_reg = self.base_cnn_coef_reg[i] , cnn_zero_order = self.base_cnn_zero_order[i],
                                     cnn_pool_type = self.base_cnn_pool_type[i], cnn_pool_size = self.base_cnn_pool_size[i], cnn_pool_stride = self.base_cnn_pool_stride[i],
                                     cnn_batch_norm = self.base_cnn_batch_norm[i], cnn_batch_norm_learn = self.base_cnn_batch_norm_learn[i],
                                     cnn_dropout_p = self.base_cnn_dropout_p[i],
                                     # Transformer parameters
                                     encoder_output_size = self.encoder_output_size, seq_type = self.base_seq_type[i],
                                     transformer_embedding_type = self.base_transformer_embedding_type[i], transformer_embedding_bias = self.base_transformer_embedding_bias[i], transformer_embedding_activation = self.base_transformer_embedding_activation[i],
                                     transformer_embedding_weight_reg = self.base_transformer_embedding_weight_reg[i], transformer_embedding_weight_norm = self.base_transformer_embedding_weight_norm[i], transformer_embedding_dropout_p = self.base_transformer_embedding_dropout_p[i],
                                     transformer_positional_encoding_type = self.base_transformer_positional_encoding_type[i],
                                     transformer_dropout1_p = self.base_transformer_dropout1_p[i], transformer_dropout2_p = self.base_transformer_dropout2_p[i], transformer_dropout3_p = self.base_transformer_dropout3_p[i],
                                     transformer_linear1_bias = self.base_transformer_linear1_bias[i], transformer_linear2_bias = self.base_transformer_linear2_bias[i],
                                     transformer_linear1_weight_reg = self.base_transformer_linear1_weight_reg[i], transformer_linear1_weight_norm = self.base_transformer_linear1_weight_norm[i],
                                     transformer_linear2_weight_reg = self.base_transformer_linear2_weight_reg[i], transformer_linear2_weight_norm = self.base_transformer_linear2_weight_norm[i],
                                     transformer_feedforward_activation = self.base_transformer_feedforward_activation[i],
                                     transformer_feedforward_degree = self.base_transformer_feedforward_degree[i], transformer_coef_init = self.base_transformer_coef_init[i], transformer_coef_train = self.base_transformer_coef_train[i], transformer_coef_reg = self.base_transformer_coef_reg[i], transformer_zero_order = self.base_transformer_zero_order[i],
                                     transformer_scale_self_attn_residual_connection = self.base_transformer_scale_self_attn_residual_connection[i],
                                     transformer_scale_cross_attn_residual_connection = self.base_transformer_scale_cross_attn_residual_connection[i],
                                     transformer_scale_feedforward_residual_connection = self.base_transformer_scale_feedforward_residual_connection[i],
                                     transformer_layer_norm = self.base_transformer_layer_norm[i],
                                     # attention parameters
                                     num_heads = self.base_num_heads[i], transformer_dim_feedforward = self.base_transformer_dim_feedforward[i],
                                     self_attn_type = self.base_self_attn_type[i], multihead_attn_type = self.base_multihead_attn_type[i],
                                     memory_is_causal = self.base_memory_is_causal[i], tgt_is_causal = self.base_tgt_is_causal[i],
                                     query_dim = self.base_query_dim[i], key_dim = self.base_key_dim[i], value_dim = self.base_value_dim[i],
                                     query_weight_reg = self.base_query_weight_reg[i], query_weight_norm = self.base_query_weight_norm[i], query_bias = self.base_query_bias[i],
                                     key_weight_reg = self.base_key_weight_reg[i], key_weight_norm = self.base_key_weight_norm[i], key_bias = self.base_key_bias[i],
                                     value_weight_reg = self.base_value_weight_reg[i], value_weight_norm = self.base_value_weight_norm[i], value_bias = self.base_value_bias[i],
                                     gen_weight_reg = self.base_gen_weight_reg[i], gen_weight_norm = self.base_gen_weight_norm[i], gen_bias = self.base_gen_bias[i],
                                     concat_weight_reg = self.base_concat_weight_reg[i], concat_weight_norm = self.base_concat_weight_norm[i], concat_bias = self.base_concat_bias[i],
                                     attn_dropout_p = self.base_attn_dropout_p[i],
                                     average_attn_weights = self.base_average_attn_weights[i],
                                     # always batch first
                                     batch_first = True,
                                     #
                                     device = self.device, dtype = self.dtype)

      if self.base_type[i] == 'lru':
        self.base_relax_init[i] = seq_base_i.base.relax_init
        self.base_relax_minmax[i] = seq_base_i.base.relax_minmax
      
      self.seq_base.append(seq_base_i)
      #
      
      # input-associated hidden layer
      if self.hidden_out_features[i] > 0:
        if self.base_hidden_size[i] > 0:
          if self.base_type[i] == 'lru':
            hidden_in_features_i = self.base_hidden_size[i]*self.base_num_filterbanks[i]
          else:
            hidden_in_features_i = (1 + int(self.base_rnn_bidirectional[i]))*self.base_hidden_size[i]
        else:
          input_size = self.input_size[i]

        hidden_layer_i = HiddenLayer(# linear transformation
                                     in_features = hidden_in_features_i, out_features = self.hidden_out_features[i],
                                     bias = self.hidden_bias[i],
                                     # activation
                                     activation = self.hidden_activation[i],
                                     # polynomial parameters
                                     degree = self.hidden_degree[i],
                                     coef_init = self.hidden_coef_init[i], coef_train = self.hidden_coef_train[i], coef_reg = self.hidden_coef_reg[i],
                                     zero_order = self.hidden_zero_order[i],
                                     # softmax parameter
                                     softmax_dim = self.hidden_softmax_dim[i],
                                     dropout_p = self.hidden_dropout_p[i],
                                     norm_type = self.norm_type,
                                     affine_norm = self.affine_norm,
                                     device = self.device, dtype = self.dtype)
      else:
        hidden_layer_i = torch.nn.Identity()

      self.hidden_layer.append(hidden_layer_i)
      #

    self.max_base_seq_len = 1 if self.process_by_step else np.max([base.output_len for base in self.seq_base])

    # interaction layer
    if self.interaction_out_features > 0:
      if sum(self.hidden_out_features) > 0:
        interaction_in_features = int(sum(self.hidden_out_features))
      else:
        interaction_in_features = 0
        for i in range(self.num_inputs):
          if self.base_type[i] in ['lstm','gru']:
            interaction_in_features += (1 + int(self.base_rnn_bidirectional[i]))*self.base_hidden_size[i]
          elif self.base_type[i] == 'lru':
            interaction_in_features += self.base_num_filterbanks[i]*self.base_hidden_size[i]
          elif self.base_type[i] == 'transformer':
            interaction_in_features += self.base_transformer_dim_feedforward[i]
          else:
            interaction_in_features += self.base_hidden_size[i]

      self.interaction_layer = HiddenLayer(# linear transformation
                                          in_features = interaction_in_features, out_features = self.interaction_out_features,
                                          bias = self.interaction_bias,
                                          # activation
                                          activation = self.interaction_activation,
                                          # polynomial parameters
                                          degree = self.interaction_degree,
                                          coef_init = self.interaction_coef_init, coef_train = self.interaction_coef_train, coef_reg = self.interaction_coef_reg,
                                          zero_order = self.interaction_zero_order,
                                          # softmax parameter
                                          softmax_dim = self.interaction_softmax_dim,
                                          dropout_p = self.interaction_dropout_p,
                                          norm_type = self.norm_type,
                                          affine_norm = self.affine_norm,
                                          device = self.device, dtype = self.dtype)
    else:
      self.interaction_layer = torch.nn.Identity()

    # modulation layer
    self.modulation_layer, self.modulation_out_features = None, 0
    if self.modulation_window_len is not None:
      if self.interaction_out_features > 0:
        modulation_in_features = self.interaction_out_features
      elif sum(self.hidden_out_features) > 0:
        modulation_in_features = sum(self.hidden_out_features)
      else:
        modulation_in_features = 0
        for i in range(self.num_inputs):
          modulation_in_features += (1 + int(self.base_rnn_bidirectional[i]))*self.base_hidden_size[i]
        else:
          modulation_in_features += self.base_transformer_dim_feedforward[i]
      #

      self.modulation_layer = ModulationLayer(window_len = self.modulation_window_len,
                                              in_features = self.modulation_in_features,
                                              associated = self.modulation_associated,
                                              legendre_degree = self.modulation_legendre_degree,
                                              chebychev_degree = self.modulation_chebychev_degree,
                                              dt = self.dt,
                                              num_freqs = self.modulation_num_freqs, freq_init = self.modulation_freq_init,  freq_train = self.modulation_freq_init,
                                              phase_init = self.modulation_phase_init, phase_train = self.modulation_phase_train,
                                              num_sigmoids = self.modulation_num_sigmoids,
                                              slope_init = self.modulation_slope_init, slope_train = self.modulation_slope_train,
                                              shift_init = self.modulation_shift_init, shift_train =  self.modulation_shift_init,
                                              weight_reg = self.modulation_weight_reg, weight_norm = self.modulation_weight_norm,
                                              zero_order = self.modulation_zero_order,
                                              bias = self.modulation_bias, pure = self.modulation_pure,
                                              norm_type = self.norm_type,
                                              affine_norm = self.affine_norm,
                                              device = self.device, dtype = self.dtype)
      self.modulation_out_features = self.modulation_layer.num_modulators
    #

    # output layer
    self.output_layer, self.Flatten = torch.nn.ModuleList([]), torch.nn.ModuleList([])
    for i in range(self.num_outputs):      
      if self.modulation_layer is not None:
        output_in_features_i = self.modulation_layer.num_modulators
      elif self.interaction_out_features > 0:
        output_in_features_i = self.interaction_out_features
      elif sum(self.hidden_out_features) > 0:
        if self.output_associated[i]:
          output_in_features_i = self.hidden_out_features[i]
        else:
          output_in_features_i = int(sum(self.hidden_out_features))
      else:
        if self.output_associated[i]:
          if self.base_type[i] in ['lstm', 'gru']:
            output_in_features_i = (1 + int(self.base_rnn_bidirectional[i]))*self.base_hidden_size[i]
          elif self.base_type[i] == 'lru':
            output_in_features_i = self.base_num_filterbanks[i]*self.base_hidden_size[i]
          elif self.base_type[i] == 'cnn':
            output_in_features_i = self.base_hidden_size[i]
          else:
            output_in_features_i = self.input_size[i]
        elif self.base_type[i] == 'identity':
          output_in_features_i = sum(self.input_size)
        else:
          output_in_features_i = 0
          for j in range(self.num_inputs):
            if self.base_type[j] in ['lstm', 'gru']:
              output_in_features_i += (1 + int(self.base_rnn_bidirectional[j]))*self.base_hidden_size[j]
            elif self.base_type[j] == 'lru':
              output_in_features_i += self.base_num_filterbanks[j]*self.base_hidden_size[j]
            else: # elif self.base_type[j] == 'cnn':
              output_in_features_i += self.base_hidden_size[j]
            
      if self.output_flatten[i] is not None:
        self.Flatten.append(torch.nn.Flatten(1, 2))

        if self.output_flatten[i] == 'time':
          # output_in_features_i = 1
          output_out_features_i = 1 
          self.output_size[i] = 1
        elif self.output_flatten[i] == 'feature': 
          # output_in_features_i = output_in_features_i * self.max_base_seq_len 
          output_out_features_i = self.output_size[i] * self.max_output_len          
          # self.output_size[i] = output_out_features_i

      else:
        self.Flatten.append(None)
        output_out_features_i = self.output_size[i]
      
      if output_size_original[i] > 0:
        
        output_layer_i = HiddenLayer(# linear transformation
                                     in_features = output_in_features_i,
                                     out_features = output_out_features_i,
                                     bias = self.output_bias[i],
                                     # activation
                                     activation = self.output_activation[i],
                                     # polynomial parameters
                                     degree = self.output_degree[i],
                                     coef_init = self.output_coef_init[i], coef_train = self.output_coef_train[i], coef_reg = self.output_coef_reg[i],
                                     zero_order = self.output_zero_order[i],
                                     # softmax parameter
                                     softmax_dim = self.output_softmax_dim[i],
                                     dropout_p = self.output_dropout_p[i],
                                     weights_to_1 = output_out_features_i == 1, # self.output_layer_w_to_1[i], # 
                                     device = self.device, dtype = self.dtype)
        
      else:
        output_layer_i = torch.nn.Identity()
        if self.output_flatten[i] == 'time':
          self.output_size[i] = 1
        elif sum(self.hidden_out_features) > 0:
          self.output_size[i] = self.hidden_out_features[i]
        elif self.interaction_out_features > 0:
          self.output_size[i] = self.interaction_out_features
        elif self.output_associated[i]:
          self.output_size[i] = (1 + int(self.base_rnn_bidirectional[i]))*self.base_hidden_size[i]
        else:
          self.output_size[i] = 0
          for j in range(self.num_inputs):
            self.output_size[i] += (1 + int(self.base_rnn_bidirectional[j]))*self.base_hidden_size[j]
        # self.output_size = (np.array(self.output_size) * self.max_output_len).tolist() if self.output_flatten else self.output_size

      self.output_layer.append(output_layer_i)

    self.total_input_size, self.total_output_size = sum(self.input_size), sum(self.output_size)

    with torch.no_grad():
      X = torch.empty((2, self.max_input_len, sum(self.input_size))).to(device = self.device,
                                                                        dtype = self.dtype)
      encoder_output = torch.empty((2, self.max_input_len, encoder_output_size)).to(X) if encoder_output_size is not None else None

      self.max_output_len = self.forward(X, encoder_output = encoder_output)[0].shape[1]

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
              input, input_window_idx = None,
              hiddens = None,
              steps = None,
              encoder_output = None):

    """
    Process the input data through the sequence model.

    Args:
        input (torch.Tensor): Input data of shape (num_samples, input_len, input_size).
        input_window_idx (list, optional): List of indices specifying the input window for each input.
        hiddens (list, optional): List of initial hidden states for each input.
        steps (int, optional): Number of processing steps.
        encoder_output (torch.Tensor, optional): Encoder output data.

    Returns:
        torch.Tensor: Processed output data.
        list: List of updated hidden states.
    """

    # Get the dimensions of the input
    num_samples, input_len, input_size = input.shape
    
    input_window_idx = [torch.arange(input_len).to(device=self.device, dtype=torch.long)
                        for _ in range(self.num_inputs)] if input_window_idx is None else input_window_idx
    
    # Initialize hidden states if not provided
    hiddens = hiddens if hiddens is not None else self.init_hiddens()

    hidden_output = []
    # Process each input in the batch individually
    for i, input_i in enumerate(input.split(self.input_size, -1)):
        
      # Determine the output feature size for the hidden layer
      if self.hidden_out_features[i] > 0:
        hidden_out_features_i = self.hidden_out_features[i]
      else:
        if self.seq_base[i].base_type in ['lstm', 'gru']:
          hidden_out_features_i = (1 + int(self.base_rnn_bidirectional[i])) * self.base_hidden_size[i]
        elif self.seq_base[i].base_type == 'lru':
          hidden_out_features_i = self.base_hidden_size[i] * self.base_num_filterbanks[i]
        else:
          hidden_out_features_i = self.base_hidden_size[i]

      # Initialize tensor for hidden layer output
      hidden_output_i = torch.zeros((num_samples, np.min([input_len, self.max_base_seq_len]), hidden_out_features_i)).to(input)

      # Generate output and updated hidden states from sequence base
      base_output_i, hiddens[i] = self.seq_base[i](input = input_i[:, -1:] if self.process_by_step
                                                   else input_i[:, input_window_idx[i]],
                                                   hiddens = hiddens[i],
                                                   encoder_output = encoder_output)
      
      # Store the output of the base layer if required
      if self.store_layer_outputs:
        self.base_layer_output[i].append(base_output_i)

      # Generate hidden layer outputs for the ith input
      hidden_output_i[:, -base_output_i.shape[1]:] = self.hidden_layer[i](base_output_i)

      hidden_output.append(hidden_output_i)

      # Store the output of the hidden layer if required
      if self.store_layer_outputs:
        self.hidden_layer_output[i].append(hidden_output_i)
    
    output_ = torch.cat(hidden_output, -1)
    
    # Generate interaction layer output
    output_ = self.interaction_layer(output_)

    # Store the output of the interaction layer if required
    if self.store_layer_outputs:
      self.interaction_layer_output.append(output_)

    # Apply modulation layer if present
    if self.modulation_layer is not None:
      output_ = self.modulation_layer(output_, steps)

      # Store the output of the modulation layer if required
      if self.store_layer_outputs:
        self.modulation_layer_output.append(output_)

    # Generate output for each output layer
    output = []
    for i in range(self.num_outputs):
      # Determine the input for the current output layer
      if self.output_associated[i]:
        output_input_i = hidden_output[i]
      else:
        output_input_i = output_
      
      # Generate output of the ith output layer      
      output_i = self.output_layer[i](output_input_i)
      
      # Flatten input for output layer if necessary
      if self.output_flatten[i] == 'time':
        output_i = self.Flatten[i](output_i).unsqueeze(2)
      elif self.output_flatten[i] == 'feature':        
        output_i = self.Flatten[i](output_i).reshape(num_samples, self.max_output_len, self.output_size[i])
        
      output.append(output_i)

      # Store the output of the output layer if required
      if self.store_layer_outputs:
        self.output_layer_output[i].append(output_i)

    # Concatenate outputs into single tensor
    output = torch.cat(output, -1)

    # Update max_output_len if necessary
    if any([flatten is not None for flatten in self.output_flatten]) & (self.max_output_len != output.shape[1]):
      self.max_output_len = output.shape[1]

    return output, hiddens

  def forward(self,
              input, steps = None,
              hiddens = None,
              target = None,
              input_window_idx = None, output_window_idx = None,
              input_mask = None, output_mask = None,
              output_input_idx = [], input_output_idx = [],
              encoder_output = None):
    
    """
    Perform forward pass through the sequence model.

    Args:
      input (torch.Tensor): Input data of shape (num_samples, input_len, input_size).
      steps (torch.Tensor, optional): Number of processing steps.
      hiddens (list, optional): List of initial hidden states for each input.
      target (torch.Tensor, optional): Target data.
      input_window_idx (list, optional): List of input window indices.
      output_window_idx (list, optional): List of output window indices.
      input_mask (torch.Tensor, optional): Input mask.
      output_mask (torch.Tensor, optional): Output mask.
      output_input_idx (list, optional): List of indices for output input.
      input_output_idx (list, optional): List of indices for input output.
      encoder_output (torch.Tensor, optional): Encoder output data.

    Returns:
      torch.Tensor: Processed output data.
      list: List of updated hidden states.
    """
    
    # Initialize lists to store layer outputs
    self.base_layer_output = [[] for _ in range(self.num_inputs)]
    self.hidden_layer_output = [[] for _ in range(self.num_inputs)]
    self.interaction_layer_output = []
    self.modulation_layer_output = []
    self.output_layer_output = [[] for _ in range(self.num_outputs)]

    # Convert inputs to the correct device
    input = input.to(device=self.device, dtype=self.dtype)
    steps = steps.to(device=self.device, dtype=torch.long) if steps is not None else None
    output_mask = output_mask.to(device=self.device, dtype=torch.long) if output_mask is not None else None

    # Get the dimensions of the input
    num_samples, input_len, input_size = input.shape

    # Prepare input window indices if not provided
    input_window_idx = [torch.arange(input_len).to(device=self.device, dtype=torch.long)
                        for _ in range(self.num_inputs)] if input_window_idx is None else input_window_idx
    output_window_idx = [torch.arange(input_len).to(device=self.device, dtype=torch.long)
                         for _ in range(self.num_outputs)] if output_window_idx is None else output_window_idx

    unique_input_window_idx = torch.cat(input_window_idx).unique()
    unique_output_window_idx = torch.cat(output_window_idx).unique()

    # Get the total output size
    total_output_size = sum(self.output_size)

    # Initiate hiddens if None
    if hiddens is None:
      hiddens = self.init_hiddens()
    else:
      for i in range(len(hiddens)):
        if (not self.base_stateful[i]):
          hiddens[i] = None

    # Process output and update hiddens
    # if 'encoder' in [base.seq_type for base in self.seq_base]: # model is an encoder

    # else: # model is a decoder
    if self.process_by_step:
      output = torch.zeros((num_samples, input_len, self.total_output_size)).to(device = self.device,
                                                                                dtype = self.dtype)
      
      output[:, :1], hiddens = self.process(input = input[:, :1],
                                            steps = steps[:, :1] if steps is not None else None,
                                            hiddens = hiddens,
                                            encoder_output = encoder_output)
      
      for n in range(1, input_len):

        input_n = input[:, n].clone()        
        if (len(output_input_idx) > 0) & (len(input_output_idx) > 0):
          input_n[:, output_input_idx] = target[:, n-1, input_output_idx] if target is not None else output[:, n-1, input_output_idx]
        
        output[:, n:(n+1)], hiddens = self.process(input = input_n.unsqueeze(1),
                                                   steps = steps[:, n:(n+1)] if steps is not None else None,
                                                   hiddens = hiddens,
                                                   encoder_output = encoder_output)
        
    else:        
      # input = torch.nn.functional.pad(input,
      #                                 (0, 0, np.max([self.max_output_len - input_len, 0]), 0),
      #                                 "constant", 0).to(input)

      output, hiddens = self.process(input = input,
                                     input_window_idx = input_window_idx,
                                     steps = steps[:, unique_input_window_idx] if steps is not None else None,
                                     hiddens = hiddens,
                                     encoder_output = encoder_output)

    # Only keep the outputs for the maximum output sequence length  
    output = output[:, -self.max_output_len:]
    
    # Apply the output mask if specified
    if output_mask is not None:
      output = output * output_mask

    # Concatenate stored layer outputs if required
    if self.store_layer_outputs:
      for i in range(self.num_inputs):
        if len(self.base_layer_output[i]) > 0:
          self.base_layer_output[i] = torch.cat(self.base_layer_output[i], 1)
        if len(self.hidden_layer_output[i]) > 0:
          self.hidden_layer_output[i] = torch.cat(self.hidden_layer_output[i], 1)
      if len(self.interaction_layer_output) > 0:
        self.interaction_layer_output = torch.cat(self.interaction_layer_output, 1)
      if len(self.modulation_layer_output) > 0:
        self.modulation_layer_output = torch.cat(self.modulation_layer_output, 1)
      for i in range(self.num_outputs):
        if len(self.output_layer_output[i]) > 0:
          self.output_layer_output[i] = torch.cat(self.output_layer_output[i], 1)

    return output, hiddens
  
  def constrain(self):

    """
    Apply constraints to the model parameters.

    Constraints are applied to different components of the model, such as
    the sequence base, hidden layers, interaction layer, and output layers.
    """

    # Apply constraints to sequence base and hidden layers for each input
    for i in range(self.num_inputs):
      if self.base_constrain[i]:
        self.seq_base[i].constrain()

      if self.hidden_constrain[i]:
        self.hidden_layer[i].constrain()

    # Apply constraints to the interaction layer if specified
    if self.interaction_constrain:
      self.interaction_layer.constrain()

    # Apply constraints to output layers for each output
    for i in range(self.num_outputs):
      if self.output_constrain[i]:
        self.output_layer[i].constrain()

  def penalize(self):

    """
    Calculate the penalty term for regularization.

    This method calculates the penalty term for each component of the model
    that is subject to regularization, such as the sequence base, hidden layers,
    interaction layer, and output layers. The penalty terms are summed up and
    returned as the total regularization loss.

    Returns:
        loss (float): Total regularization loss due to penalty terms.
    """

    loss = 0

    # Calculate penalty terms and accumulate them for sequence base and hidden layers of each input
    for i in range(self.num_inputs):
        if self.base_penalize[i]:
            loss += self.seq_base[i].penalize()

        if self.hidden_penalize[i]:
            loss += self.hidden_layer[i].penalize()

    # Calculate penalty term for interaction layer if specified
    if self.interaction_penalize:
        loss += self.interaction_layer.penalize()

    # Calculate penalty terms and accumulate them for output layers of each output
    for i in range(self.num_outputs):
        if self.output_penalize[i]:
            loss += self.output_layer[i].penalize()

    return loss

  def generate_impulse_response(self, seq_len):
    """
    Generate impulse responses for each input feature.

    This method generates impulse responses for each input feature of the model.
    It creates an impulse input signal for each feature, passes it through the
    sequence base and hidden layer, and collects the corresponding impulse responses.

    Args:
        seq_len (int): Length of the sequence for which impulse responses are generated.

    Returns:
        impulse_response (list): A list containing impulse responses for each input feature.
                                Each entry in the list is a tensor representing the
                                impulse response for a specific input feature.
    """

    with torch.no_grad():
        impulse_response = [None for _ in range(self.num_inputs)]

        # Generate impulse response for each input and each feature
        for i in range(self.num_inputs):
            impulse_response[i] = [None for _ in range(self.input_size[i])]
            for f in range(self.input_size[i]):
                # Create impulse input signal for the current feature
                impulse_i = torch.zeros((1, seq_len, self.input_size[i])).to(device=self.device,
                                                                              dtype=self.dtype)
                impulse_i[0, 0, f] = 1.

                # Pass the impulse input through sequence base and hidden layer
                base_output_if, _ = self.seq_base[i](input=impulse_i)
                impulse_response[i][f] = self.hidden_layer[i].F[0](base_output_if)[0]

    return impulse_response

  def predict(self,
            input, steps=None,
            hiddens=None,
            encoder_output=None,
            input_window_idx=None, output_window_idx=None,
            input_mask=None, output_mask=None,
            output_input_idx=[], input_output_idx=[],
            output_transforms=None):
    """
    Perform prediction using the model.

    This method performs prediction using the model. It takes input data, optionally steps data,
    and other parameters to generate predictions. It returns the prediction results and associated time steps.

    Args:
        input (Tensor): Input data tensor.
        steps (Tensor, optional): Steps data tensor. Default is None.
        hiddens (list of Tensors, optional): Initial hidden state tensors. Default is None.
        encoder_output (Tensor, optional): Encoder output tensor. Default is None.
        input_window_idx (list of Tensors, optional): Indices for input windows. Default is None.
        output_window_idx (list of Tensors, optional): Indices for output windows. Default is None.
        input_mask (Tensor, optional): Input mask tensor. Default is None.
        output_mask (Tensor, optional): Output mask tensor. Default is None.
        output_input_idx (list, optional): Indices for output input. Default is an empty list.
        input_output_idx (list, optional): Indices for input output. Default is an empty list.
        output_transforms (list of Transform objects, optional): Output transforms for prediction. Default is None.

    Returns:
        prediction (Tensor): Prediction results tensor.
        prediction_time (Tensor): Associated time steps tensor.
    """

    # Clone input and steps if provided
    input = input.clone()
    steps = steps.clone() if steps is not None else None

    num_samples, input_len, input_size = input.shape

    with torch.no_grad():
        # Generate prediction using the model
        prediction, hiddens = self.forward(input=input,
                                           steps=steps,
                                           hiddens=hiddens,
                                           input_window_idx=input_window_idx, output_window_idx=output_window_idx,
                                           input_mask=input_mask, output_mask=output_mask,
                                           output_input_idx=output_input_idx,
                                           input_output_idx=input_output_idx,
                                           encoder_output=encoder_output)

    # Extract prediction steps and compute associated time steps
    prediction_steps = steps[:, -self.max_output_len:] if steps is not None else None
    prediction_time = prediction_steps * self.dt if prediction_steps is not None else None

    # Apply output transforms if provided
    if output_transforms:
        for sampled_idx in range(num_samples):
            j = 0
            for i in range(self.num_outputs):
                prediction[sampled_idx, :, j:(j + self.output_size[i])] = output_transforms[i].inverse_transform(prediction[sampled_idx, :, j:(j + self.output_size[i])])
                j += self.output_size[i]

    return prediction, prediction_time

  def forecast(self,
              input, steps=None,
              hiddens=None,
              num_forecast_steps=1,
              encoder_output=None,
              input_window_idx=None, output_window_idx=None,
              input_mask=None, output_mask=None,
              output_input_idx=[], input_output_idx=[],
              output_transforms=None):
    """
    Perform forecasting using the model.

    Args:
        input (Tensor): Input data tensor.
        steps (Tensor, optional): Steps data tensor. Default is None.
        hiddens (list of Tensors, optional): Initial hidden state tensors. Default is None.
        num_forecast_steps (int): Number of forecast steps. Default is 1.
        encoder_output (Tensor, optional): Encoder output tensor. Default is None.
        input_window_idx (list of Tensors, optional): Indices for input windows. Default is None.
        output_window_idx (list of Tensors, optional): Indices for output windows. Default is None.
        input_mask (Tensor, optional): Input mask tensor. Default is None.
        output_mask (Tensor, optional): Output mask tensor. Default is None.
        output_input_idx (list, optional): Indices for output input. Default is an empty list.
        input_output_idx (list, optional): Indices for input output. Default is an empty list.
        output_transforms (list of Transform objects, optional): Output transforms for forecasting. Default is None.

    Returns:
        forecast (Tensor): Forecast results tensor.
        forecast_time (Tensor): Associated time steps tensor.
    """

    # Clone input and steps if provided
    input = input.clone()
    steps = steps.clone() if steps is not None else None

    with torch.no_grad():
      num_samples, input_len, input_size = input.shape

      # Initialize forecast and forecast steps tensors
      forecast = torch.empty((num_samples, 0, self.total_output_size)).to(device=self.device,
                                                                        dtype=self.dtype)
      if steps is not None:
        forecast_steps = torch.empty((num_samples, 0)).to(steps)
      else:
        forecast_steps = None

      # Calculate forecast length based on window indices
      if (input_window_idx is not None) & (output_window_idx is not None):
        max_input_window_idx = np.max([idx.max().cpu() for idx in input_window_idx])
        max_output_window_idx = np.max([idx.max().cpu() for idx in output_window_idx])
        forecast_len = np.max([1, max_output_window_idx - max_input_window_idx])
      else:
        forecast_len = 1

      # Perform initial prediction using the model
      prediction, hiddens = self.forward(input = input,
                                          steps = steps,
                                          hiddens = hiddens,
                                          input_window_idx = input_window_idx,
                                          output_window_idx = output_window_idx,
                                          encoder_output = encoder_output,
                                          input_output_idx = input_output_idx,
                                          output_input_idx = output_input_idx)

      # Concatenate initial prediction to forecast
      forecast = torch.cat((forecast, prediction[:, -forecast_len:]), 1)
      if steps is not None:
          forecast_steps = torch.cat((forecast_steps, steps[:, -forecast_len:]), 1)
          steps += forecast_len

      # Continue forecasting iteratively
      while forecast.shape[1] < (forecast_len + num_forecast_steps):

          # Prepare input for next forecasting step
          input_ar = torch.zeros((num_samples, forecast_len, input_size)).to(input)
          if (len(input_output_idx) > 0) & (len(output_input_idx) > 0):
              input_ar[..., output_input_idx] = forecast[:, -forecast_len:, input_output_idx]

          # Concatenate input for next forecasting step
          input = torch.cat((input[:, forecast_len:], input_ar), 1)

          # Perform forecasting step using the model
          prediction, hiddens = self(input=input,
                                     steps=steps,
                                     hiddens=hiddens,
                                     encoder_output=encoder_output,
                                     input_output_idx=input_output_idx,
                                     output_input_idx=output_input_idx)

          # Concatenate current prediction to forecast
          forecast = torch.cat((forecast, prediction[:, -forecast_len:]), 1)
          if steps is not None:
              forecast_steps = torch.cat((forecast_steps, steps[:, -forecast_len:]), 1)
              steps += forecast_len

    # Apply output transforms if provided
    if output_transforms:
        for sampled_idx in range(num_samples):
            j = 0
            for i in range(self.num_outputs):
                forecast[sampled_idx, :, j:(j + self.output_size[i])] = output_transforms[i].inverse_transform(
                    forecast[sampled_idx, :, j:(j + self.output_size[i])])
                j += self.output_size[i]

    # Extract the forecast for the desired number of forecast steps
    forecast = forecast[:, forecast_len:][:, :num_forecast_steps]
    forecast_steps = forecast_steps[:, forecast_len:][:, :num_forecast_steps] if forecast_steps is not None else None

    # Calculate forecast time steps
    forecast_time = forecast_steps * self.dt if forecast_steps is not None else None
    
    return forecast, forecast_time
