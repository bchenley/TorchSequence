import torch
import numpy as np

from ts_src.HiddenLayer import Polynomial
from ts_src.Polynomial import Polynomial

class Attention(torch.nn.MultiheadAttention):
  '''
  Custom attention layer based on the torch.nn.MultiheadAttention module.
  This layer supports different types of attention mechanisms: dot, general, and concat.

  Args:
      embed_dim (int): The input embedding dimension.
      num_heads (int): The number of attention heads.
      query_dim (int, optional): The query embedding dimension. Defaults to None (same as embed_dim).
      key_dim (int, optional): The key embedding dimension. Defaults to None (same as embed_dim).
      value_dim (int, optional): The value embedding dimension. Defaults to None (same as embed_dim).
      attn_type (str, optional): The attention type. Options: 'dot', 'general', 'concat'. Defaults to 'dot'.
      query_weight_reg (List[float], optional): The regularization weights for the query projection layer. Defaults to [0.001, 1].
      query_weight_norm (float, optional): The normalization type for the query projection layer. Defaults to 2.
      query_bias (bool, optional): Whether to include bias in the query projection layer. Defaults to False.
      key_weight_reg (List[float], optional): The regularization weights for the key projection layer. Defaults to [0.001, 1].
      key_weight_norm (float, optional): The normalization type for the key projection layer. Defaults to 2.
      key_bias (bool, optional): Whether to include bias in the key projection layer. Defaults to False.
      value_weight_reg (List[float], optional): The regularization weights for the value projection layer. Defaults to [0.001, 1].
      value_weight_norm (float, optional): The normalization type for the value projection layer. Defaults to 2.
      value_bias (bool, optional): Whether to include bias in the value projection layer. Defaults to False.
      gen_weight_reg (List[float], optional): The regularization weights for the generation weights (concat type). Defaults to [0.001, 1].
      gen_weight_norm (float, optional): The normalization type for the generation weights (concat type). Defaults to 2.
      gen_bias (bool, optional): Whether to include bias in the generation weights (concat type). Defaults to False.
      concat_weight_reg (List[float], optional): The regularization weights for the concatenation layer (concat type). Defaults to [0.001, 1].
      concat_weight_norm (float, optional): The normalization type for the concatenation layer (concat type). Defaults to 2.
      concat_bias (bool, optional): Whether to include bias in the concatenation layer (concat type). Defaults to False.
      average_attn_weights (bool, optional): Whether to average the attention weights across heads. Defaults to False.
      is_causal (bool, optional): Whether the attention is causal (supports autoregressive property). Defaults to False.
      dropout_p (float, optional): The dropout probability. Defaults to 0.0.
      device (str, optional): The device for the computation. Defaults to 'cpu'.
      dtype (torch.dtype, optional): The data type. Defaults to torch.float32.
  '''

  def __init__(self,
               embed_dim, num_heads=1,
               query_dim=None, key_dim=None, value_dim=None,
               attn_type="dot",
               query_weight_reg=[0.001, 1], query_weight_norm=2, query_bias=False,
               key_weight_reg=[0.001, 1], key_weight_norm=2, key_bias=False,
               value_weight_reg=[0.001, 1], value_weight_norm=2, value_bias=False,
               gen_weight_reg=[0.001, 1], gen_weight_norm=2, gen_bias=False,
               concat_weight_reg=[0.001, 1], concat_weight_norm=2, concat_bias=False,
               average_attn_weights=False,
               is_causal=False,
               dropout_p=0.0,
               device="cpu",
               dtype=torch.float32):

      super(Attention, self).__init__(embed_dim = embed_dim, 
                                      num_heads = num_heads)

      locals_ = locals().copy()

      for arg in locals_:
        if arg != 'self':
          setattr(self, arg, locals_[arg])
          
      self.dropout = torch.nn.Dropout(self.dropout_p)
                 
      # Choose the appropriate score function based on the attention type
      if self.attn_type == "dot":
          self.score_fn = self.dot_fn
      elif self.attn_type == "general":
          self.score_fn = self.general_fn
      elif self.attn_type == "concat":
          self.score_fn = self.concat_fn

      self.query_dim = self.query_dim or self.embed_dim
      self.key_dim = self.key_dim or self.embed_dim
      self.value_dim = self.value_dim or self.embed_dim

      self.query_blocks = torch.nn.ModuleList([])
      self.key_blocks = torch.nn.ModuleList([])
      self.value_blocks = torch.nn.ModuleList([])
      self.gen_blocks = torch.nn.ModuleList([])
      self.concat_blocks = torch.nn.ModuleList([])

      self.head_dims = np.round(self.embed_dim / self.num_heads).astype(int).repeat(self.num_heads - 1).tolist()
      self.head_dims += [int(self.embed_dim - np.sum(self.head_dims))]

      for dim in self.head_dims:
        self.query_blocks.append(HiddenLayer(in_features = self.embed_dim,
                                             out_features = dim,
                                             bias = self.query_bias,
                                             activation = "identity",
                                             weight_reg = self.query_weight_reg,
                                             weight_norm = self.query_weight_norm,
                                             device = self.device,
                                             dtype = self.dtype))
        self.key_blocks.append(HiddenLayer(in_features = self.embed_dim,
                                           out_features = dim,
                                           bias = self.key_bias,
                                           activation = "identity",
                                           weight_reg = self.key_weight_reg,
                                           weight_norm = self.key_weight_norm,
                                           device = self.device,
                                           dtype = self.dtype))
        self.value_blocks.append(HiddenLayer(in_features = self.embed_dim,
                                              out_features = dim,
                                              bias = self.value_bias,
                                              activation = "identity",
                                              weight_reg = self.value_weight_reg,
                                              weight_norm = self.value_weight_norm,
                                              device = self.device,
                                              dtype = self.dtype))

        if self.attn_type == "general":
          self.gen_blocks.append(HiddenLayer(in_features = [dim, dim],
                                             out_features = 1,
                                             bias = self.gen_bias,
                                             activation = "identity",
                                             weight_reg = self.gen_weight_reg,
                                             weight_norm = self.gen_weight_norm,
                                             device = self.device,
                                             dtype = self.dtype))

        if self.attn_type == "concat":
          self.concat_blocks.append(torch.nn.Sequential(*[HiddenLayer(in_features = 2 * dim,
                                                                      out_features = dim,
                                                                      bias = self.concat_bias,
                                                                      activation = "tanh",
                                                                      weight_reg = self.concat_weight_reg,
                                                                      weight_norm = self.concat_weight_norm,
                                                                      device = self.device,
                                                                      dtype = self.dtype),
                                                      HiddenLayer(in_features = dim,
                                                                  out_features = 1,
                                                                  bias = self.concat_bias,
                                                                  activation = "identity",
                                                                  weight_reg = self.concat_weight_reg,
                                                                  weight_norm = self.concat_weight_norm,
                                                                  device = self.device,
                                                                  dtype = self.dtype)]))

  def dot_fn(self, query, key, block_idx):
    '''
    Compute the dot-product attention score between query and key.

    Args:
        query (torch.Tensor): The query tensor of shape (num_samples, query_len, query_dim).
        key (torch.Tensor): The key tensor of shape (num_samples, key_len, key_dim).
        block_idx (int): The index of the attention block.

    Returns:
        torch.Tensor: The attention score tensor of shape (num_samples, query_len, key_len).
    '''
    score = (torch.bmm(query, key.transpose(-2, -1)) / torch.math.sqrt(query.shape[-1])).transpose(-1, -2)
    return score

  def general_fn(self, query, key, block_idx):
    '''
    Compute the general attention score between query and key.

    Args:
        query (torch.Tensor): The query tensor of shape (num_samples, query_len, query_dim).
        key (torch.Tensor): The key tensor of shape (num_samples, key_len, key_dim).
        block_idx (int): The index of the attention block.

    Returns:
        torch.Tensor: The attention score tensor of shape (num_samples, query_len, key_len).
    '''
    if query.shape[1] == 1:
        query = query.repeat(1, key.shape[1], 1)

    score = self.gen_blocks[block_idx]((query, key))

    return score

  def concat_fn(self, query, key, block_idx):
    '''
    Compute the concat attention score between query and key.

    Args:
        query (torch.Tensor): The query tensor of shape (num_samples, query_len, query_dim).
        key (torch.Tensor): The key tensor of shape (num_samples, key_len, key_dim).
        block_idx (int): The index of the attention block.

    Returns:
        torch.Tensor: The attention score tensor of shape (num_samples, query_len, key_len).
    '''
    if query.shape[1] == 1:
        query = query.repeat(1, key.shape[1], 1)

    score = self.concat_blocks[block_idx](torch.cat((query, key), -1))
    return score

  def forward(self, query, key, value, attn_mask=None):
    '''
    Perform the forward pass of the attention layer.

    Args:
        query (torch.Tensor): The query tensor of shape (num_samples, query_len, query_dim).
        key (torch.Tensor): The key tensor of shape (num_samples, key_len, key_dim).
        value (torch.Tensor): The value tensor of shape (num_samples, value_len, value_dim).
        attn_mask (torch.Tensor, optional): The attention mask tensor of shape (query_len, key_len)
            or (num_samples, num_heads, query_len, key_len). Defaults to None.

    Returns:
        torch.Tensor: The output tensor of shape (num_samples, query_len, value_dim).
    '''
    num_samples, query_len, query_dim = query.shape
    _, key_len, key_dim = key.shape
    _, value_len, value_dim = value.shape

    ones = torch.ones((query_len, key_len), device=self.device, dtype=torch.bool)

    attn_mask = ones.tril(diagonal=0).transpose(-2, -1) if self.is_causal else ones
    attn_mask = attn_mask.to(query).masked_fill(~attn_mask, -float('inf')) if attn_mask.dtype == torch.bool else attn_mask

    output, weight = [], []
    for block_idx, (query_block, key_block, value_block) in enumerate(zip(self.query_blocks, self.key_blocks, self.value_blocks)):

      query_h, key_h, value_h = query_block(query), key_block(key), value_block(value)

      score_h = self.score_fn(query_h, key_h, block_idx)

      weight_h = torch.softmax(score_h + attn_mask, dim=1)

      output_h = torch.bmm(weight_h.transpose(-2, -1), value_h)

      weight.append(weight_h)
      output.append(output_h)

    output = self.dropout(torch.cat(output, -1))
    weight = torch.cat(weight, 1)

    if self.average_attn_weights:
      weight = weight.mean(1)

    self.weight = weight

    return output

  def penalize(self):
    '''
    Compute the regularization loss for the attention layer.

    Returns:
        torch.Tensor: The regularization loss.
    '''
    loss = 0
    for name, param in self.named_parameters():
      if 'weight' in name:
        if 'query' in name:
          loss += self.query_weight_reg[0] * torch.norm(param, p=self.query_weight_reg[1]) * int(param.requires_grad)
        elif 'key' in name:
          loss += self.key_weight_reg[0] * torch.norm(param, p=self.key_weight_reg[1]) * int(param.requires_grad)
        elif 'value' in name:
          loss += self.value_weight_reg[0] * torch.norm(param, p=self.value_weight_reg[1]) * int(param.requires_grad)
        elif 'gen' in name:
          loss += self.gen_weight_reg[0] * torch.norm(param, p=self.gen_weight_reg[1]) * int(param.requires_grad)
        elif 'concat' in name:
          loss += self.concat_weight_reg[0] * torch.norm(param, p=self.concat_weight_reg[1]) * int(param.requires_grad)

    return loss
