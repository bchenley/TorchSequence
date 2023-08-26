import torch 

from Attention import Attention
from Polynomial import Polynomial

class TransformerEncoderLayer(torch.nn.TransformerEncoderLayer):
    '''
    Customized Transformer Encoder Layer with optional modifications.

    Args:
      d_model (int): The input and output feature dimension.
      nhead (int, optional): The number of attention heads. Defaults to 1.
      dim_feedforward (int, optional): The hidden dimension of the feedforward network. Defaults to 2048.
      self_attn_type (str, optional): The type of self-attention. Choices: 'dot', 'general', 'concat'. Defaults to 'dot'.
      is_causal (bool, optional): Whether to use causal self-attention. Defaults to False.
      query_weight_reg (list[float], optional): Regularization weights for the query weights. Defaults to [0.001, 1].
      query_weight_norm (int, optional): Norm type for the query weights. Defaults to 2.
      query_bias (bool, optional): Whether to use bias in the query weights. Defaults to False.
      key_weight_reg (list[float], optional): Regularization weights for the key weights. Defaults to [0.001, 1].
      key_weight_norm (int, optional): Norm type for the key weights. Defaults to 2.
      key_bias (bool, optional): Whether to use bias in the key weights. Defaults to False.
      value_weight_reg (list[float], optional): Regularization weights for the value weights. Defaults to [0.001, 1].
      value_weight_norm (int, optional): Norm type for the value weights. Defaults to 2.
      value_bias (bool, optional): Whether to use bias in the value weights. Defaults to False.
      gen_weight_reg (list[float], optional): Regularization weights for the generator weights. Defaults to [0.001, 1].
      gen_weight_norm (int, optional): Norm type for the generator weights. Defaults to 2.
      gen_bias (bool, optional): Whether to use bias in the generator weights. Defaults to False.
      concat_weight_reg (list[float], optional): Regularization weights for the concatenator weights. Defaults to [0.001, 1].
      concat_weight_norm (int, optional): Norm type for the concatenator weights. Defaults to 2.
      concat_bias (bool, optional): Whether to use bias in the concatenator weights. Defaults to False.
      average_attn_weights (bool, optional): Whether to average the attention weights. Defaults to False.
      dropout_p (float, optional): Dropout probability for the attention and feedforward layers. Defaults to 0.0.
      dropout1_p (float, optional): Dropout probability for the first dropout layer in the feedforward network. Defaults to 0.0.
      dropout2_p (float, optional): Dropout probability for the second dropout layer in the feedforward network. Defaults to 0.0.
      linear1_bias (bool, optional): Whether to use bias in the first linear layer of the feedforward network. Defaults to False.
      linear2_bias (bool, optional): Whether to use bias in the second linear layer of the feedforward network. Defaults to False.
      linear1_weight_reg (list[float], optional): Regularization weights for the first linear layer weights. Defaults to [0.001, 1].
      linear1_weight_norm (int, optional): Norm type for the first linear layer weights. Defaults to 2.
      linear2_weight_reg (list[float], optional): Regularization weights for the second linear layer weights. Defaults to [0.001, 1].
      linear2_weight_norm (int, optional): Norm type for the second linear layer weights. Defaults to 2.
      feedforward_activation (str, optional): The activation function in the feedforward network. Choices: 'identity', 'relu', 'gelu', 'polynomial'. Defaults to 'relu'.
      degree (int, optional): The degree of the polynomial activation function. Only applicable when feedforward_activation='polynomial'. Defaults to 2.
      coef_init (torch.Tensor, optional): The initial coefficients for the polynomial activation function. Only applicable when feedforward_activation='polynomial'. Defaults to None.
      coef_train (bool, optional): Whether to train the coefficients for the polynomial activation function. Only applicable when feedforward_activation='polynomial'. Defaults to True.
      coef_reg (list[float], optional): Regularization weights for the polynomial coefficients. Only applicable when feedforward_activation='polynomial'. Defaults to [0.001, 1.].
      zero_order (bool, optional): Whether to include the zero-order term in the polynomial activation function. Only applicable when feedforward_activation='polynomial'. Defaults to False.
      scale_self_attn_residual_connection (bool, optional): Whether to scale the self-attention residual connection. Defaults to False.
      scale_feedforward_residual_connection (bool, optional): Whether to scale the feedforward residual connection. Defaults to False.
      device (str, optional): The device to run the layer on. Defaults to 'cpu'.
      dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
    '''

    def __init__(self,
                d_model, nhead=1, dim_feedforward=2048,
                self_attn_type='dot',
                is_causal=False,
                query_weight_reg=[0.001, 1], query_weight_norm=2, query_bias=False,
                key_weight_reg=[0.001, 1], key_weight_norm=2, key_bias=False,
                value_weight_reg=[0.001, 1], value_weight_norm=2, value_bias=False,
                gen_weight_reg=[0.001, 1], gen_weight_norm=2, gen_bias=False,
                concat_weight_reg=[0.001, 1], concat_weight_norm=2, concat_bias=False,
                average_attn_weights=False,
                dropout_p=0.0, dropout1_p=0.0, dropout2_p=0.0,
                linear1_bias=False, linear2_bias=False,
                linear1_weight_reg=[0.001, 1], linear1_weight_norm=2,
                linear2_weight_reg=[0.001, 1], linear2_weight_norm=2,
                feedforward_activation='relu',
                degree=2,
                coef_init=None, coef_train=True, coef_reg=[0.001, 1.],
                zero_order=False,
                scale_self_attn_residual_connection=False,
                scale_feedforward_residual_connection=False,
                device='cpu', dtype=torch.float32):

        super(TransformerEncoderLayer, self).__init__(d_model=d_model,
                                                      nhead=nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      device=device,
                                                      dtype=dtype)

        locals_ = locals().copy()

        for arg in locals_:
          if arg != 'self':
            setattr(self, arg, locals_[arg])
          
        self.dropout.p = dropout_p

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
                                   is_causal = self.is_causal,
                                   dropout_p = self.dropout_p,
                                   device = self.device,
                                   dtype = self.dtype)

        self.dropout1.p = self.dropout1_p
        self.dropout2.p = self.dropout2_p

        if self.feedforward_activation == 'identity':
          self.activation = torch.nn.Identity()
          self.linear2 = torch.nn.Identity()
          self.norm2 = torch.nn.Identity()
          self.dropout2 = torch.nn.Identity()

        elif self.feedforward_activation == 'relu':
          self.activation = torch.nn.ReLU()
        elif self.feedforward_activation == 'gelu':
          self.activation = torch.nn.GELU()
        elif self.feedforward_activation == 'polynomial':
          self.activation = Polynomial(in_features = self.dim_feedforward,
                                       degree = self.degree,
                                       coef_init = self.coef_init,
                                       coef_train = self.coef_train,
                                       coef_reg = self.coef_reg,
                                       zero_order = self.zero_order,
                                       device = self.device,
                                       dtype = self.dtype)

        if not self.linear1_bias:
          self.linear1.bias = None

        self.linear1_weight_reg = self.linear1_weight_reg
        self.linear1_weight_norm = self.linear1_weight_norm

        if not isinstance(self.linear2, torch.nn.Identity):
          if not self.linear2_bias:
            self.linear2.bias = None
        
        self.linear2_weight_reg = self.linear2_weight_reg
        self.linear2_weight_norm = self.linear2_weight_norm

        if self.scale_self_attn_residual_connection:
            self.self_attn_residual_scaler = torch.nn.Linear(in_features = self.d_model, out_features = 1).weight.squeeze().to(device = self.device, dtype = self.dtype)
        else:
            self.self_attn_residual_scaler = torch.ones((self.d_model,)).to(device = self.device, dtype = self.dtype)

        if self.scale_feedforward_residual_connection:
            self.feedforward_residual_scaler = torch.nn.Linear(in_features = self.d_model, out_features = 1).weight.squeeze().to(device = self.device, dtype = self.dtype)
        else:
            self.feedforward_residual_scaler = torch.ones((self.d_model,)).to(device = self.device, dtype = self.dtype)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
      '''
      Forward pass of the transformer encoder layer.

      Args:
          src (torch.Tensor): The input sequence of shape (seq_len, batch_size, d_model).
          src_mask (torch.Tensor, optional): The mask to apply to the source sequence. Defaults to None.
          src_key_padding_mask (torch.Tensor, optional): The padding mask for the source sequence. Defaults to None.
          is_causal (bool, optional): Whether to use causal self-attention. Defaults to False.

      Returns:
          torch.Tensor: The output sequence of shape (seq_len, batch_size, d_model).
      '''

      # Generate self-attn output (dropout applied inside) and add residual connection (scale if desired)
      src = self.self_attn(src, src, src, src_mask) + self.self_attn_residual_scaler * src

      # Normalize self-attn sub-layer
      src = self.norm1(src)

      # Generate feedforward output and add residual connection (scale if desired)
      src = self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(src))))) + self.feedforward_residual_scaler * src

      src = self.norm2(src)

      return src

    def penalize(self):
      '''
      Calculate the regularization loss for the transformer encoder layer.

      Returns:
          torch.Tensor: The regularization loss.
      '''
      loss = 0
      if self.self_attn is not None:
        loss += self.self_attn.penalize()
      loss += (self.linear1_weight_reg[0]
               * torch.norm(self.linear1.weight, p=self.linear1_weight_reg[1])
               * int(self.linear1.weight.requires_grad))
      loss += (self.linear2_weight_reg[0]
               * torch.norm(self.linear2.weight, p=self.linear2_weight_reg[1])
               * int(self.linear2.weight.requires_grad))

      return loss
