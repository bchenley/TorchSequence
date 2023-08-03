import torch 

from ts_src.LRU import LRU
from ts_src.HiddenLayer import HiddenLayer
from ts_src.Embedding import Embedding
from ts_src.Attention import Attention
from ts_src.PositionalEncoding import PositionalEncoding
from ts_src.TransformerEncoderLayer import TransformerEncoderLayer
from ts_src.TransformerDecoderLayer import TransformerDecoderLayer
from ts_src.CNN1D import CNN1D

class SequenceModelBase(torch.nn.Module):
  '''
  Base class for sequence models.

  Args:
    input_size (int): The number of expected features in the input.
    hidden_size (int): The number of features in the hidden state/output.
    input_len (int, optional): The length of the input sequence. Default is None.
    output_len (int, optional): The length of the output sequence.
    base_type (str): The type of the base model. Options: 'gru', 'lstm', 'lru', 'cnn', 'transformer'.
    num_layers (int, optional): Number of recurrent layers. Default is 1.
    encoder_bias (bool, optional): Whether to include a bias term in the encoder block. Default is False.
    decoder_bias (bool, optional): Whether to include a bias term in the decoder block. Default is False.
    rnn_bias (bool, optional): If False, then the layer does not use bias weights. Default is True.
    rnn_dropout_p (float, optional): Dropout probability for the base model. Default is 0.
    rnn_bidirectional (bool, optional): If True, becomes a bidirectional RNN. Default is False.
    rnn_attn (bool, optional): Whether to apply attention mechanism on RNN outputs. Default is False.
    rnn_weight_reg (list, optional): Regularization settings for RNN weights. Default is [0.001, 1].
    rnn_weight_norm (float, optional): Norm type for RNN weights. Default is None.
    relax_init (list, optional): Initial relaxation values for LRU. Default is [0.5].
    relax_train (bool, optional): Whether to train relaxation values for LRU. Default is True.
    relax_minmax (list, optional): Minimum and maximum relaxation values for LRU. Default is [0.1, 0.9].
    num_filterbanks (int, optional): Number of filterbanks for LRU. Default is 1.
    cnn_kernel_size (tuple, optional): Size of the convolving kernel for CNN. Default is (1,).
    cnn_kernel_stride (tuple, optional): Stride of the convolution for CNN. Default is (1,).
    cnn_padding (tuple, optional): Zero-padding added to both sides of the input for CNN. Default is (0,).
    cnn_dilation (tuple, optional): Spacing between kernel elements for CNN. Default is (1,).
    cnn_groups (int, optional): Number of blocked connections from input channels to output channels for CNN. Default is 1.
    cnn_bias (bool, optional): If False, then the layer does not use bias weights. Default is False.
    encoder_output_size (int, optional): The size of the output from the encoder block. Default is None.
    seq_type (str, optional): Type of the sequence. Options: 'encoder', 'decoder'. Default is 'encoder'.
    transformer_embedding_type (str, optional): Type of embedding for Transformer. Default is 'time'.
    transformer_embedding_bias (bool, optional): Whether to include a bias term in the embedding for Transformer. Default is False.
    transformer_embedding_activation (str, optional): Activation function for Transformer embedding. Options: 'identity', 'relu', 'gelu'. Default is 'identity'.
    transformer_embedding_weight_reg (list, optional): Regularization settings for Transformer embedding weights. Default is [0.001, 1].
    transformer_embedding_weight_norm (float, optional): Norm type for Transformer embedding weights. Default is 2.
    transformer_embedding_dropout_p (float, optional): Dropout probability for Transformer embedding. Default is 0.0.
    transformer_positional_encoding_type (str, optional): Type of positional encoding for Transformer. Options: 'absolute'.
    transformer_dropout1_p (float, optional): Dropout probability for the first dropout layer in Transformer. Default is 0.
    transformer_dropout2_p (float, optional): Dropout probability for the second dropout layer in Transformer. Default is 0.
    transformer_dropout3_p (float, optional): Dropout probability for the third dropout layer in Transformer. Default is 0.
    transformer_linear1_bias (bool, optional): Whether to include a bias term in the first linear layer in Transformer. Default is False.
    transformer_linear2_bias (bool, optional): Whether to include a bias term in the second linear layer in Transformer. Default is False.
    transformer_linear1_weight_reg (list, optional): Regularization settings for the weights of the first linear layer in Transformer. Default is [0.001, 1].
    transformer_linear1_weight_norm (float, optional): Norm type for the weights of the first linear layer in Transformer. Default is 2.
    transformer_linear2_weight_reg (list, optional): Regularization settings for the weights of the second linear layer in Transformer. Default is [0.001, 1].
    transformer_linear2_weight_norm (float, optional): Norm type for the weights of the second linear layer in Transformer. Default is 2.
    transformer_feedforward_activation (str, optional): Activation function for the feedforward layer in Transformer. Options: 'relu'. Default is 'relu'.
    transformer_feedforward_degree (int, optional): Degree of the polynomial activation function for the feedforward layer in Transformer. Default is 2.
    transformer_coef_init (None or float, optional): Initial value for the coefficients of the polynomial activation function in Transformer. Default is None.
    transformer_coef_train (bool, optional): Whether to train the coefficients of the polynomial activation function in Transformer. Default is True.
    transformer_coef_reg (list, optional): Regularization settings for the coefficients of the polynomial activation function in Transformer. Default is [0.001, 1.].
    transformer_zero_order (bool, optional): Whether to include the zero-order term in the polynomial activation function in Transformer. Default is False.
    transformer_scale_self_attn_residual_connection (bool, optional): Whether to scale the residual connection in the self-attention sub-layer of Transformer. Default is False.
    transformer_scale_cross_attn_residual_connection (bool, optional): Whether to scale the residual connection in the cross-attention sub-layer of Transformer. Default is False.
    transformer_scale_feedforward_residual_connection (bool, optional): Whether to scale the residual connection in the feedforward sub-layer of Transformer. Default is False.
    transformer_layer_norm (bool, optional): Whether to include layer normalization in Transformer layers. Default is True.
    num_heads (int, optional): Number of attention heads in Transformer. Default is 1.
    transformer_dim_feedforward (int, optional): Dimension of the feedforward layer in Transformer. Default is 2048.
    self_attn_type (str, optional): Type of self-attention in Transformer. Options: 'dot'. Default is 'dot'.
    multihead_attn_type (str, optional): Type of multihead attention in Transformer. Options: 'dot'. Default is 'dot'.
    memory_is_causal (bool, optional): Whether the memory sequence is causal in Transformer. Default is True.
    tgt_is_causal (bool, optional): Whether the target sequence is causal in Transformer. Default is False.
    query_dim (None or int, optional): The dimension of query in attention mechanism. Default is None.
    key_dim (None or int, optional): The dimension of key in attention mechanism. Default is None.
    value_dim (None or int, optional): The dimension of value in attention mechanism. Default is None.
    query_weight_reg (list, optional): Regularization settings for the query weight in attention mechanism. Default is [0.001, 1].
    query_weight_norm (float, optional): Norm type for the query weight in attention mechanism. Default is 2.
    query_bias (bool, optional): Whether to include a bias term in the query weight in attention mechanism. Default is False.
    key_weight_reg (list, optional): Regularization settings for the key weight in attention mechanism. Default is [0.001, 1].
    key_weight_norm (float, optional): Norm type for the key weight in attention mechanism. Default is 2.
    key_bias (bool, optional): Whether to include a bias term in the key weight in attention mechanism. Default is False.
    value_weight_reg (list, optional): Regularization settings for the value weight in attention mechanism. Default is [0.001, 1].
    value_weight_norm (float, optional): Norm type for the value weight in attention mechanism. Default is 2.
    value_bias (bool, optional): Whether to include a bias term in the value weight in attention mechanism. Default is False.
    gen_weight_reg (list, optional): Regularization settings for the generator weight in attention mechanism. Default is [0.001, 1].
    gen_weight_norm (float, optional): Norm type for the generator weight in attention mechanism. Default is 2.
    gen_bias (bool, optional): Whether to include a bias term in the generator weight in attention mechanism. Default is False.
    concat_weight_reg (list, optional): Regularization settings for the concatenation weight in attention mechanism. Default is [0.001, 1].
    concat_weight_norm (float, optional): Norm type for the concatenation weight in attention mechanism. Default is 2.
    concat_bias (bool, optional): Whether to include a bias term in the concatenation weight in attention mechanism. Default is False.
    attn_dropout_p (float, optional): Dropout probability for attention mechanism. Default is 0.
    average_attn_weights (bool, optional): Whether to average attention weights. Default is False.
    batch_first (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default is True.
    device (str, optional): The device to run the model on. Default is 'cpu'.
    dtype (torch.dtype, optional): The desired data type of the model's parameters. Default is torch.float32.
  '''

  def __init__(self,
              input_size, hidden_size, input_len,
              base_type='gru', num_layers=1,
              stateful = False,
              encoder_bias=False, decoder_bias=False,
              rnn_bias = True,
              rnn_dropout_p = 0,
              rnn_bidirectional = False,
              rnn_attn=False,
              rnn_weight_reg=[0.001, 1], rnn_weight_norm=None,
              relax_init=[0.5], relax_train=True, relax_minmax=[0.1, 0.9], num_filterbanks=1,
              cnn_out_channels = None, 
              cnn_pad_front = False,
              cnn_kernel_size = [(1,)], cnn_kernel_stride = [(1,)], cnn_padding = [(0,)], cnn_dilation = [(1,)], cnn_groups = [1],
              cnn_bias = [False], cnn_pool_type = [None], cnn_pool_size = [(2,)], cnn_pool_stride = [(0,)],
              cnn_batch_norm = False, cnn_batch_norm_learn = False,
              encoder_output_size = None, seq_type = 'encoder',
              transformer_embedding_type='time', transformer_embedding_bias=False,
              transformer_embedding_activation='identity',
              transformer_embedding_weight_reg=[0.001, 1], transformer_embedding_weight_norm=2,
              transformer_embedding_dropout_p=0.0,
              transformer_positional_encoding_type='absolute',
              transformer_dropout1_p=0., transformer_dropout2_p=0., transformer_dropout3_p=0.,
              transformer_linear1_bias=False, transformer_linear2_bias=False,
              transformer_linear1_weight_reg=[0.001, 1], transformer_linear1_weight_norm=2,
              transformer_linear2_weight_reg=[0.001, 1], transformer_linear2_weight_norm=2,
              transformer_feedforward_activation='relu',
              transformer_feedforward_degree=2, transformer_coef_init=None, transformer_coef_train=True,
              transformer_coef_reg=[0.001, 1.], transformer_zero_order=False,
              transformer_scale_self_attn_residual_connection=False,
              transformer_scale_cross_attn_residual_connection=False,
              transformer_scale_feedforward_residual_connection=False,
              transformer_layer_norm=True,
              num_heads=1, transformer_dim_feedforward=2048, 
              self_attn_type='dot', multihead_attn_type='dot',
              memory_is_causal=True, tgt_is_causal=False,
              query_dim=None, key_dim=None, value_dim=None,
              query_weight_reg=[0.001, 1], query_weight_norm=2, query_bias=False,
              key_weight_reg=[0.001, 1], key_weight_norm=2, key_bias=False,
              value_weight_reg=[0.001, 1], value_weight_norm=2, value_bias=False,
              gen_weight_reg=[0.001, 1], gen_weight_norm=2, gen_bias=False,
              concat_weight_reg=[0.001, 1], concat_weight_norm=2, concat_bias=False,
              attn_dropout_p=0.,
              average_attn_weights=False,
              batch_first=True,
              device='cpu', dtype=torch.float32):
    super(SequenceModelBase, self).__init__()

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])
        
    # self.to(device = self.device, 
    #         dtype = self.dtype)

    self.name = f"{self.base_type}{self.num_layers}"

    self.positional_encoding = None
    self.encoder_block = None
    if self.base_type == 'identity':
      self.base = torch.nn.Identity() 
    elif self.base_type == 'gru':
      self.base = torch.nn.GRU(input_size = self.input_size,
                                hidden_size = self.hidden_size,
                                num_layers = self.num_layers,
                                bias = self.rnn_bias,
                                dropout = self.rnn_dropout_p,
                                bidirectional = self.rnn_bidirectional,
                                device = self.device, dtype = self.dtype,
                                batch_first = True)
    elif self.base_type == 'lstm':
      self.base = torch.nn.LSTM(input_size = self.input_size,
                                hidden_size = self.hidden_size,
                                num_layers = self.num_layers,
                                bias = self.rnn_bias,
                                dropout = self.rnn_dropout_p,
                                bidirectional = self.rnn_bidirectional,
                                device = self.device, dtype = self.dtype,
                                batch_first = True)
    elif self.base_type == 'lru':
      self.base = LRU(input_size = self.input_size, hidden_size = self.hidden_size,
                      bias = self.rnn_bias,
                      relax_init = self.relax_init, relax_train = self.relax_train, relax_minmax = self.relax_minmax,
                      device = self.device, dtype = self.dtype)
    elif self.base_type == 'cnn':
      self.base = CNN1D(in_channels = self.input_size, 
                        out_channels = self.cnn_out_channels, 
                        pad_front = self.cnn_pad_front,
                        input_len = input_len,
                        kernel_size = self.cnn_kernel_size, 
                        kernel_stride = self.cnn_kernel_stride, 
                        padding = self.cnn_padding, 
                        dilation = self.cnn_dilation, 
                        groups = self.cnn_groups,
                        bias = self.cnn_bias, 
                        pool_type = self.cnn_pool_type, 
                        pool_size = self.cnn_pool_size,
                        pool_stride = self.cnn_pool_stride,
                        batch_norm = cnn_batch_norm, 
                        batch_norm_learn = cnn_batch_norm_learn,
                        device = self.device, dtype = self.dtype)

    elif self.base_type == 'transformer':
      embedding = Embedding(num_embeddings = self.input_size,
                            embedding_dim = self.hidden_size,
                            embedding_type = self.transformer_embedding_type,
                            bias = self.transformer_embedding_bias,
                            activation = self.transformer_embedding_activation,
                            weight_reg = self.transformer_embedding_weight_reg,
                            weight_norm = self.transformer_embedding_weight_norm,
                            dropout_p = self.transformer_embedding_dropout_p,
                            device = self.device, dtype = self.dtype)

      positional_encoding = PositionalEncoding(dim = self.hidden_size, input_len = self.input_len,
                                                encoding_type = self.transformer_positional_encoding_type,
                                                device = self.device, dtype = self.dtype)

      self.base = torch.nn.ModuleList([torch.nn.Sequential(*[embedding, positional_encoding])])

      if self.seq_type == 'encoder':
          self.base.append(torch.nn.TransformerEncoder(TransformerEncoderLayer(d_model = self.hidden_size,
                                                                              nhead = self.num_heads,
                                                                              dim_feedforward = self.transformer_dim_feedforward,
                                                                              self_attn_type = self.self_attn_type,
                                                                              is_causal = self.memory_is_causal,
                                                                              query_weight_reg = self.query_weight_reg,
                                                                              query_weight_norm = self.query_weight_norm,
                                                                              query_bias = self.query_bias,
                                                                              key_weight_reg = self.key_weight_reg,
                                                                              key_weight_norm = self.key_weight_norm,
                                                                              key_bias = self.key_bias,
                                                                              value_weight_reg = self.value_weight_reg,
                                                                              value_weight_norm = self.value_weight_norm,
                                                                              value_bias = self.value_bias,
                                                                              gen_weight_reg = self.gen_weight_reg,
                                                                              gen_weight_norm = self.gen_weight_norm,
                                                                              gen_bias = self.gen_bias,
                                                                              concat_weight_reg = self.concat_weight_reg,
                                                                              concat_weight_norm = self.concat_weight_norm,
                                                                              concat_bias = self.concat_bias,
                                                                              average_attn_weights = self.average_attn_weights,
                                                                              dropout_p = self.attn_dropout_p,
                                                                              dropout1_p = self.transformer_dropout1_p,
                                                                              dropout2_p = self.transformer_dropout2_p,
                                                                              linear1_weight_reg = self.transformer_linear1_weight_reg,
                                                                              linear1_weight_norm = self.transformer_linear1_weight_norm,
                                                                              linear2_weight_reg = self.transformer_linear2_weight_reg,
                                                                              linear2_weight_norm = self.transformer_linear2_weight_norm,
                                                                              linear1_bias = self.transformer_linear1_bias,
                                                                              linear2_bias = self.transformer_linear2_bias,
                                                                              feedforward_activation = self.transformer_feedforward_activation,
                                                                              degree = self.transformer_feedforward_degree,
                                                                              coef_init = self.transformer_coef_init,
                                                                              coef_train = self.transformer_coef_train,
                                                                              coef_reg = self.transformer_coef_reg,
                                                                              zero_order = self.transformer_zero_order,
                                                                              scale_self_attn_residual_connection = self.transformer_scale_self_attn_residual_connection,
                                                                              scale_feedforward_residual_connection = self.transformer_scale_feedforward_residual_connection,
                                                                              device = self.device, dtype = self.dtype),
                                                  num_layers = self.num_layers))

      elif self.seq_type == 'decoder':
        self.base.append(torch.nn.TransformerDecoder(TransformerDecoderLayer(d_model = self.hidden_size,
                                                                            nhead = self.num_heads,
                                                                            dim_feedforward = self.transformer_dim_feedforward,
                                                                            self_attn_type = self.self_attn_type,
                                                                            memory_is_causal = self.memory_is_causal,
                                                                            tgt_is_causal = self.tgt_is_causal,
                                                                            query_weight_reg = self.query_weight_reg,
                                                                            query_weight_norm = self.query_weight_norm,
                                                                            query_bias = self.query_bias,
                                                                            key_weight_reg = self.key_weight_reg,
                                                                            key_weight_norm = self.key_weight_norm,
                                                                            key_bias = self.key_bias,
                                                                            value_weight_reg = self.value_weight_reg,
                                                                            value_weight_norm = self.value_weight_norm,
                                                                            value_bias = self.value_bias,
                                                                            gen_weight_reg = self.gen_weight_reg,
                                                                            gen_weight_norm = self.gen_weight_norm,
                                                                            gen_bias = self.gen_bias,
                                                                            concat_weight_reg = self.concat_weight_reg,
                                                                            concat_weight_norm = self.concat_weight_norm,
                                                                            concat_bias = self.concat_bias,
                                                                            average_attn_weights = self.average_attn_weights,
                                                                            dropout_p = self.attn_dropout_p,
                                                                            dropout1_p = self.transformer_dropout1_p,
                                                                            dropout2_p = self.transformer_dropout2_p,
                                                                            dropout3_p = self.transformer_dropout3_p,
                                                                            linear1_weight_reg = self.transformer_linear1_weight_reg,
                                                                            linear1_weight_norm = self.transformer_linear1_weight_norm,
                                                                            linear2_weight_reg = self.transformer_linear2_weight_reg,
                                                                            linear2_weight_norm = self.transformer_linear2_weight_norm,
                                                                            linear1_bias = self.transformer_linear1_bias,
                                                                            linear2_bias = self.transformer_linear2_bias,
                                                                            feedforward_activation = self.transformer_feedforward_activation,
                                                                            degree = self.transformer_feedforward_degree,
                                                                            coef_init = self.transformer_coef_init,
                                                                            coef_train = self.transformer_coef_train,
                                                                            coef_reg = self.transformer_coef_reg,
                                                                            zero_order = self.transformer_zero_order,
                                                                            scale_self_attn_residual_connection = self.transformer_scale_self_attn_residual_connection,
                                                                            scale_cross_attn_residual_connection = self.transformer_scale_cross_attn_residual_connection,
                                                                            scale_feedforward_residual_connection = self.transformer_scale_feedforward_residual_connection,
                                                                            device = self.device, dtype = self.dtype),
                                                num_layers = self.num_layers))

        if (self.encoder_output_size != self.hidden_size):
            self.encoder_block = HiddenLayer(in_features = self.encoder_output_size,
                                              out_features = self.hidden_size,
                                              activation = 'identity',
                                              bias = self.encoder_bias,
                                              device = self.device, dtype = self.dtype)

        self.base[1].norm = None if not self.transformer_layer_norm else self.base[1].norm

      else:
        raise ValueError(f"'{self.base_type}' is not a valid value. `base_type` must be 'gru', 'lstm', 'lru', 'cnn', or 'transformer'.")

    self.attn_mechanism, self.decoder_block = None, None
    if self.rnn_attn:
        self.attn_mechanism = Attention(embed_dim = self.hidden_size,
                                        num_heads = self.num_heads,
                                        query_dim = self.query_dim, key_dim = self.key_dim, value_dim = self.value_dim,
                                        attn_type = self.multihead_attn_type,
                                        query_weight_reg = self.query_weight_reg, query_weight_norm = self.query_weight_norm,
                                        query_bias = self.query_bias,
                                        key_weight_reg = self.key_weight_reg, key_weight_norm = self.key_weight_norm, key_bias = self.key_bias,
                                        value_weight_reg = self.value_weight_reg, value_weight_norm = self.value_weight_norm, value_bias = self.value_bias,
                                        is_causal = self.tgt_is_causal, dropout_p = self.attn_dropout_p,
                                        device = self.device, dtype = self.dtype)

        if (self.encoder_output_size != self.hidden_size * (1 + self.rnn_bidirectional)):
            self.encoder_block = HiddenLayer(in_features = self.encoder_output_size,
                                        out_features = (self.hidden_size * (1 + self.rnn_bidirectional) if self.base_type in ('lstm', 'gru') else len(self.relax_init)) * self.hidden_size * (1 + self.rnn_bidirectional),
                                        activation = 'identity',
                                        bias = self.encoder_bias,
                                        device = self.device, dtype = self.dtype)

        self.decoder_block = HiddenLayer(in_features = 2 * self.hidden_size,
                                         out_features = self.hidden_size,
                                         activation = 'identity',
                                         bias = self.decoder_bias,
                                         device = self.device, dtype = self.dtype)

    with torch.no_grad():            
      X = torch.empty((1, self.input_len, input_size)).to(device = self.device,
                                                          dtype = self.dtype)
      encoder_output = torch.empty((1,self.input_len,encoder_output_size)).to(device = self.device,
                                                                              dtype = self.dtype) if encoder_output_size is not None else None
      
      self.output_len = self.forward(X, encoder_output = encoder_output)[0].shape[1]
                
  def init_hiddens(self, num_samples):
    '''
    Initialize hidden states for the base model.

    Args:
        num_samples (int): The number of samples in the input.

    Returns:
        hiddens (list or torch.Tensor): Initialized hidden states.
    '''
    if self.base_type == 'lru':
        hiddens = None # torch.zeros((self.base.num_filterbanks, num_samples, self.base.hidden_size)).to(device=self.device,
                                                                                                    # dtype=self.dtype)
    else:
        if self.base_type == 'lstm':
            hiddens = None # [torch.zeros((self.base.num_layers*(1+int(self.base.bidirectional)), num_samples, self.base.hidden_size)).to(device=self.device,
                                                                                                                                   # dtype=self.dtype)] * 2                                                                                                                                   
        else:
            hiddens = None # torch.zeros((self.base.num_layers*(1+int(self.base.bidirectional)), num_samples, self.base.hidden_size)).to(device=self.device,
                                                                                                                                  # dtype=self.dtype)

    return hiddens

  def forward(self, input, hiddens=None, encoder_output=None, mask=None):
    '''
    Forward pass of the sequence model.

    Args:
        input (torch.Tensor): Input tensor of shape (num_samples, input_len, input_size).
        hiddens (list or torch.Tensor, optional): Hidden states of the base model. Default is None.
        encoder_output (torch.Tensor, optional): Output from the encoder block. Default is None.
        mask (torch.Tensor, optional): Mask tensor for attention mechanism. Default is None.

    Returns:
        output (torch.Tensor): Output tensor of shape (num_samples, input_len, output_size).
        hiddens (list or torch.Tensor): Updated hidden states of the base model.
    '''
    num_samples, input_len, input_size = input.shape

    if (hiddens is None) | (not self.stateful): # (self.base_type in ['lru', 'lstm', 'gru']):
        hiddens = self.init_hiddens(num_samples)
    
    if self.encoder_block is not None:
        encoder_output = self.encoder_block(encoder_output)

    if self.base_type == 'identity':
        output, hiddens = input, hiddens
    elif self.base_type in ['lru', 'lstm', 'gru']:        
        output, hiddens = self.base(input, hiddens)

        output = output.reshape(num_samples, input_len, -1)

        if self.rnn_attn:
            # Pass encoder output and base output (context) to generate attn output and weights
            attn_output = self.attn_mechanism(query=output[:, -1:],
                                              key=encoder_output,
                                              value=encoder_output)

            # Ensure attn_output has the same length as the base output
            if attn_output.shape[1] == 1:
                attn_output = attn_output.repeat(1, output.shape[1], 1)

            # Combine attn output and base output, then pass result to the decoder block to generate the new base output
            output = self.decoder_block(torch.cat((attn_output, output), -1))

    elif self.base_type == 'cnn':
        output = self.base(input)
    elif self.base_type == 'transformer':
        input_embedding_pe = self.base[0](input)

        output = self.base[1](tgt=input_embedding_pe, memory=encoder_output) if self.seq_type == 'decoder' \
            else self.base[1](src=input_embedding_pe, mask=mask)

    return output, hiddens

  def constrain(self):
    '''
    Apply constraints to the model.

    This method applies constraints specific to each base model type.
    '''
    if self.base_type == 'lru':
        self.base.clamp_relax()
    elif self.weight_norm is not None:
        for name, param in self.named_parameters():
            if 'weight' in name:
                param = torch.nn.functional.normalize(param, p=self.rnn_weight_norm, dim=1).contiguous()

  def penalize(self):
    '''
    Compute the penalty for regularization.

    Returns:
        loss (torch.Tensor): Regularization loss.
    '''
    loss = 0
    if self.base_type == 'transformer':
        loss += self.base[0].penalize()  # embedding penalty
        loss += sum(layer.penalize() for layer in self.base[1])  # transformer layer penalties
    else:
        for name, param in self.named_parameters():
            if 'weight' in param:
                loss += self.rnn_weight_reg[0] * torch.norm(param, p=self.rnn_weight_reg[1]) * int(
                    param.requires_grad)

    return loss
