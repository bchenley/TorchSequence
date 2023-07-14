import torch

class TransformerDecoderLayer(torch.nn.TransformerDecoderLayer):
  '''
  Transformer Decoder Layer module that extends `torch.nn.TransformerDecoderLayer`.

  Args:
    d_model (int): The number of expected features in the input.
    nhead (int, optional): The number of heads in the multihead attention models. Defaults to 1.
    dim_feedforward (int, optional): The dimension of the feedforward network model. Defaults to 2048.
    self_attn_type (str, optional): The self-attention type. Defaults to 'dot'.
    multihead_attn_type (str, optional): The multihead attention type. Defaults to 'dot'.
    memory_is_causal (bool, optional): Whether the memory sequence is causal. Defaults to False.
    tgt_is_causal (bool, optional): Whether the target sequence is causal. Defaults to True.
    query_weight_reg (list, optional): Regularization parameters for query weight. Defaults to [0.001, 1].
    query_weight_norm (int, optional): Norm type for query weight regularization. Defaults to 2.
    query_bias (bool, optional): Whether to include bias in query weight. Defaults to False.
    key_weight_reg (list, optional): Regularization parameters for key weight. Defaults to [0.001, 1].
    key_weight_norm (int, optional): Norm type for key weight regularization. Defaults to 2.
    key_bias (bool, optional): Whether to include bias in key weight. Defaults to False.
    value_weight_reg (list, optional): Regularization parameters for value weight. Defaults to [0.001, 1].
    value_weight_norm (int, optional): Norm type for value weight regularization. Defaults to 2.
    value_bias (bool, optional): Whether to include bias in value weight. Defaults to False.
    gen_weight_reg (list, optional): Regularization parameters for generation weight. Defaults to [0.001, 1].
    gen_weight_norm (int, optional): Norm type for generation weight regularization. Defaults to 2.
    concat_weight_reg (list, optional): Regularization parameters for concatenation weight. Defaults to [0.001, 1].
    concat_weight_norm (int, optional): Norm type for concatenation weight regularization. Defaults to 2.
    concat_bias (bool, optional): Whether to include bias in concatenation weight. Defaults to False.
    average_attn_weights (bool, optional): Whether to average the attention weights. Defaults to False.
    dropout_p (float, optional): Probability of an element to be zeroed. Defaults to 0.
    dropout1_p (float, optional): Probability of an element of the first dropout layer to be zeroed. Defaults to 0.
    dropout2_p (float, optional): Probability of an element of the second dropout layer to be zeroed. Defaults to 0.
    dropout3_p (float, optional): Probability of an element of the third dropout layer to be zeroed. Defaults to 0.
    linear1_bias (bool, optional): Whether to include bias in the first linear layer. Defaults to False.
    linear2_bias (bool, optional): Whether to include bias in the second linear layer. Defaults to False.
    linear1_weight_reg (list, optional): Regularization parameters for the first linear layer weight. Defaults to [0.001, 1].
    linear1_weight_norm (int, optional): Norm type for the first linear layer weight regularization. Defaults to 2.
    linear2_weight_reg (list, optional): Regularization parameters for the second linear layer weight. Defaults to [0.001, 1].
    linear2_weight_norm (int, optional): Norm type for the second linear layer weight regularization. Defaults to 2.
    feedforward_activation (str, optional): Type of activation function for the feedforward network. Defaults to 'relu'.
    degree (int, optional): Degree of the polynomial activation function. Defaults to 2.
    coef_init (torch.Tensor, optional): Initial coefficients for the polynomial activation function. Defaults to None.
    coef_train (bool, optional): Whether to train the coefficients of the polynomial activation function. Defaults to True.
    coef_reg (list, optional): Regularization parameters for the polynomial activation function coefficients. Defaults to [0.001, 1.].
    zero_order (bool, optional): Whether to include the zero-order term in the polynomial activation function. Defaults to False.
    scale_self_attn_residual_connection (bool, optional): Whether to scale the self-attention residual connection. Defaults to False.
    scale_cross_attn_residual_connection (bool, optional): Whether to scale the cross-attention residual connection. Defaults to False.
    scale_feedforward_residual_connection (bool, optional): Whether to scale the feedforward residual connection. Defaults to False.
    device (str, optional): Device on which to allocate tensors. Defaults to 'cpu'.
    dtype (torch.dtype, optional): Desired data type of the tensor. Defaults to torch.float32.
  '''

  def __init__(self,
               d_model, nhead=1, dim_feedforward=2048,
               self_attn_type="dot", multihead_attn_type="dot",
               memory_is_causal=False, tgt_is_causal=True,
               query_weight_reg=[0.001, 1], query_weight_norm=2, query_bias=False,
               key_weight_reg=[0.001, 1], key_weight_norm=2, key_bias=False,
               value_weight_reg=[0.001, 1], value_weight_norm=2, value_bias=False,
               gen_weight_reg=[0.001, 1], gen_weight_norm=2, gen_bias = False,
               concat_weight_reg=[0.001, 1], concat_weight_norm=2, concat_bias=False,
               average_attn_weights=False,
               dropout_p=0.0, dropout1_p=0.0, dropout2_p=0.0, dropout3_p=0.0,
               linear1_bias=False, linear2_bias=False,
               linear1_weight_reg=[0.001, 1], linear1_weight_norm=2,
               linear2_weight_reg=[0.001, 1], linear2_weight_norm=2,
               feedforward_activation="relu",
               degree=2,
               coef_init=None, coef_train=True, coef_reg=[0.001, 1.],
               zero_order=False,
               scale_self_attn_residual_connection=False,
               scale_cross_attn_residual_connection=False,
               scale_feedforward_residual_connection=False,
               device="cpu", dtype=torch.float32):

      super(TransformerDecoderLayer, self).__init__(d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    device=device,
                                                    dtype=dtype)

      locals_ = locals().copy()

      for arg in locals_:
        if arg != 'self':
          setattr(self, arg, locals_[arg])
        
      self.dropout.p = self.dropout_p

      self.self_attn = Attention(embed_dim = self.d_model,
                                 num_heads = self.nhead,
                                 attn_type = self.self_attn_type,
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
                                 is_causal = self.memory_is_causal,
                                 dropout_p = self.dropout_p,
                                 device = self.device,
                                 dtype = self.dtype)

      self.multihead_attn = Attention(embed_dim = self.d_model,
                                      num_heads = self.nhead,
                                      attn_type = self.multihead_attn_type,
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
                                      is_causal = self.tgt_is_causal,
                                      dropout_p = self.dropout1_p,
                                      device = self.device,
                                      dtype = self.dtype)

      self.dropout2.p = self.dropout2_p
      self.dropout3.p = self.dropout3_p

      if self.feedforward_activation == "identity":
        self.activation = torch.nn.Identity()
        self.linear2 = torch.nn.Identity()
        self.norm3 = torch.nn.Identity()
        self.dropout3 = torch.nn.Identity()
      elif self.feedforward_activation == "relu":
        self.activation = torch.nn.ReLU()
      elif self.feedforward_activation == "gelu":
        self.activation = torch.nn.GELU()
      elif self.feedforward_activation == "polynomial":
        self.activation = Polynomial(in_features = self.dim_feedforward,
                                    degree = self.degree,
                                    coef_init = self.coef_init,
                                    coef_train = self.coef_train,
                                    coef_reg = self.coef_reg,
                                    zero_order = self.zero_order,
                                    device = self.device,
                                    dtype = self.dtype)

      self.linear1.bias = None if not self.linear1_bias else self.linear1.bias
      self.linear1_weight_reg, self.linear1_weight_norm = self.linear1_weight_reg, self.linear1_weight_norm

      if not isinstance(self.linear2, torch.nn.Identity):
          self.linear2.bias = None if not self.linear2_bias else self.linear2.bias
          self.linear2_weight_reg, self.linear2_weight_norm = self.linear2_weight_reg, self.linear2_weight_norm

      self.self_attn_residual_scaler = (torch.nn.Linear(in_features = self.d_model, out_features = 1).weight.squeeze().to(device = self.device, dtype = self.dtype)
                                        if self.scale_self_attn_residual_connection
                                        else torch.ones((self.d_model,)).to(device = self.device, dtype = self.dtype))

      self.cross_attn_residual_scaler = (torch.nn.Linear(in_features = self.d_model, out_features = 1).weight.squeeze().to(device = self.device, dtype = self.dtype)
                                          if self.scale_cross_attn_residual_connection
                                          else torch.ones((self.d_model,)).to(device = self.device, dtype = self.dtype))

      self.feedforward_residual_scaler = (torch.nn.Linear(in_features = self.d_model, out_features = 1).weight.squeeze().to(device = self.device, dtype = self.dtype)
                                          if self.scale_feedforward_residual_connection
                                          else torch.ones((self.d_model,)).to(device = self.device, dtype = self.dtype))

  def forward(self,
              tgt, memory,
              tgt_mask=None, memory_mask=None,
              tgt_key_padding_mask=None, memory_key_padding_mask=None):
      '''
      Forward pass of the Transformer Decoder Layer.

      Args:
          tgt (torch.Tensor): The input to the decoder layer of shape `(target_sequence_length, batch_size, d_model)`.
          memory (torch.Tensor): The output of the encoder layer of shape `(input_sequence_length, batch_size, d_model)`.
          tgt_mask (torch.Tensor, optional): Mask applied to the target sequence. Defaults to None.
          memory_mask (torch.Tensor, optional): Mask applied to the memory sequence. Defaults to None.
          tgt_key_padding_mask (torch.Tensor, optional): Mask applied to the target keys. Defaults to None.
          memory_key_padding_mask (torch.Tensor, optional): Mask applied to the memory keys. Defaults to None.

      Returns:
          torch.Tensor: The output of the decoder layer of shape `(target_sequence_length, batch_size, d_model)`.
      '''

      tgt = self.self_attn(tgt, tgt, tgt, tgt_mask) + self.self_attn_residual_scaler * tgt
      tgt = self.norm1(tgt)
      tgt = self.multihead_attn(tgt, memory, memory, memory_mask) + self.cross_attn_residual_scaler * tgt
      tgt = self.norm2(tgt)
      tgt = self.dropout3(self.linear2(self.dropout2(self.activation(self.linear1(tgt))))) + self.feedforward_residual_scaler * tgt
      tgt = self.norm3(tgt)

      return tgt

  def penalize(self):
      '''
      Compute the regularization loss for the decoder layer.

      Returns:
          torch.Tensor: The regularization loss.
      '''
      loss = 0
      loss += self.self_attn.penalize()
      loss += self.multihead_attn.penalize()
      loss += self.linear1_weight_reg[0] * torch.norm(self.linear1.weight, p=self.linear1_weight_reg[1]) * int(self.linear1.weight.requires_grad)
      loss += self.linear2_weight_reg[0] * torch.norm(self.linear2.weight, p=self.linear2_weight_reg[1]) * int(self.linear2.weight.requires_grad)

      return loss
