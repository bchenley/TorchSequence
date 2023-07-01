class Embedding(torch.nn.Module):
    '''
    Embedding layer that maps input tokens to continuous vectors.

    Args:
        num_embeddings (int): Number of unique tokens in the input vocabulary.
        embedding_dim (int): Dimensionality of the embedding vectors.
        embedding_type (str, optional): Type of embedding to use. Supported types: 'time', 'category'.
                                        Defaults to 'time'.
        bias (bool, optional): Whether to include a bias term in the embedding layer. Defaults to False.
        activation (str, optional): Activation function to apply to the embedding. Defaults to 'identity'.
        weight_reg (List[float], optional): Regularization terms for the embedding weights.
                                             Defaults to [0.001, 1].
        weight_norm (float, optional): Order of the normalization applied to the embedding weights.
                                       Defaults to 2.
        dropout_p (float, optional): Dropout probability to apply to the embedding layer. Defaults to 0.0.
        device (str, optional): Device on which the embedding layer is allocated. Defaults to 'cpu'.
        dtype (torch.dtype, optional): Data type of the embedding layer. Defaults to torch.float32.
    '''

    def __init__(self,
                 num_embeddings, embedding_dim, embedding_type='time',
                 bias=False, activation='identity',
                 weight_reg=[0.001, 1], weight_norm=2,
                 dropout_p=0.0,
                 device='cpu', dtype=torch.float32):
      super(Embedding, self).__init__()

      # Check the type of embedding
      if embedding_type == 'time':
          # Time-based embedding using HiddenLayer
          embedding = HiddenLayer(in_features=num_embeddings,
                                  out_features=embedding_dim,
                                  bias=bias,
                                  activation=activation,
                                  weight_reg=weight_reg, weight_norm=weight_norm,
                                  dropout_p=dropout_p,
                                  device=device, dtype=dtype)
      elif embedding_type == 'category':
          # Category-based embedding using torch.nn.Embedding
          embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
      else:
          raise ValueError(f"Unsupported embedding type: {embedding_type}")

      self.embedding = embedding
      self.embedding_type = embedding_type

    def forward(self, input, input_mask=None):
      '''
      Forward pass of the embedding layer.

      Args:
          input (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
          input_mask (torch.Tensor): Input mask tensor of shape (batch_size, sequence_length)
                                      or None if no mask is applied.

      Returns:
          torch.Tensor: Embedded input tensor of shape (batch_size, sequence_length, embedding_dim).
      '''

      # Apply input mask if provided
      input = input*input_mask if input_mask is not None else input

      # Embed the input
      input_embedding = self.embedding(input)

      return input_embedding
