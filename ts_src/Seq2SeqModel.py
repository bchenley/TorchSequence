import torch
import numpy as np

from ts_src import HiddenLayer as HiddenLayer

class Seq2SeqModel(torch.nn.Module):
  
  '''
  Sequence-to-Sequence Model that consists of an encoder and a decoder.

  Args:
      encoder (torch.nn.Module): The encoder module.
      decoder (torch.nn.Module): The decoder module.
      learn_decoder_input (bool, optional): Whether to learn the decoder's initial input. Defaults to False.
      learn_decoder_hiddens (bool, optional): Whether to learn the decoder's hidden states. Defaults to False.
      enc2dec_bias (bool, optional): Whether to use bias in the encoder-to-decoder mappings. Defaults to True.
      enc2dec_hiddens_bias (bool, optional): Whether to use bias in the encoder-to-decoder hidden state mappings. Defaults to True.
      enc2dec_dropout_p (float, optional): Dropout probability for the encoder-to-decoder mappings. Defaults to 0.
      enc2dec_hiddens_dropout_p (float, optional): Dropout probability for the encoder-to-decoder hidden state mappings. Defaults to 0.
      device (str, optional): Device to run the model on (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
      dtype (torch.dtype, optional): Data type of the model parameters. Defaults to torch.float32.
  '''
    
  def __init__(self,
              encoder, decoder,
              learn_decoder_input=False, learn_decoder_hiddens=False,
              enc2dec_bias=True, enc2dec_hiddens_bias=True,
              enc2dec_dropout_p=0., enc2dec_hiddens_dropout_p=0.,
              enc_out_as_dec_in = False,
              device='cpu', dtype=torch.float32):

    super(Seq2SeqModel, self).__init__()

    locals_ = locals().copy()

    for arg in locals_:
      if arg != 'self':
        setattr(self, arg, locals_[arg])

    self.num_inputs, self.num_outputs = self.encoder.num_inputs, self.decoder.num_outputs
    self.input_size, self.output_size = self.encoder.input_size, self.decoder.output_size
    self.base_type = self.encoder.base_type
                
    self.enc2dec_input_block = None
    if self.learn_decoder_input:
      
      self.enc2dec_input_block = HiddenLayer(in_features = sum(self.encoder.input_size),
                                             out_features = sum(self.decoder.input_size),
                                             bias = self.enc2dec_bias,
                                             activation = 'identity',
                                             dropout_p = self.enc2dec_dropout_p,
                                             device = self.device,
                                             dtype = self.dtype)
      
    self.enc2dec_hiddens_block = None
    
    if self.learn_decoder_hiddens:
      if any(type_ in ['gru', 'lstm', 'lru'] for type_ in self.encoder.base_type):
        total_encoder_hidden_size = 0
        for i in range(self.encoder.num_inputs):
          if self.encoder.base_type[i] in ['gru', 'lstm', 'lru']:
            total_encoder_hidden_size += (1 + int(self.encoder.base_type[i] == 'lstm')) * self.encoder.base_hidden_size[i]      
      else:
        total_encoder_hidden_size = sum(self.encoder.output_size)
      
      self.decoder_hidden_size = []
      for i in range(self.num_inputs):
        self.decoder_hidden_size.append(0)        
        self.decoder_hidden_size[i] = (1+int(self.decoder.base_type[i] == 'lstm')) * (1 + int(self.decoder.base_rnn_bidirectional[i])) * self.decoder.base_hidden_size[i]

      self.enc2dec_hiddens_block = HiddenLayer(in_features = total_encoder_hidden_size,
                                               out_features = sum(self.decoder_hidden_size),
                                               bias = self.enc2dec_hiddens_bias,
                                               activation = 'identity',
                                               dropout_p = self.enc2dec_hiddens_dropout_p,
                                               device = self.device,
                                               dtype = self.dtype)
  
  def forward(self,
              input,
              steps=None,
              hiddens=None,
              input_mask=None, output_mask=None,
              output_input_idx=[], input_output_idx=[],
              encoder_output=None,
              target=None,
              output_window_idx=None):

    '''
    Forward pass of the Seq2SeqModel.

    Args:
        input (torch.Tensor): Input tensor of shape (num_samples, input_len, input_size).
        steps (torch.Tensor, optional): Tensor indicating the number of steps to process for each sample. Shape: (num_samples, input_len). Defaults to None.
        hiddens (list, optional): List of initial hidden states for the encoder. Defaults to None.
        input_mask (torch.Tensor, optional): Mask tensor for the input. Shape: (num_samples, input_len). Defaults to None.
        output_mask (torch.Tensor, optional): Mask tensor for the output. Shape: (num_samples, output_len). Defaults to None.
        output_input_idx (list, optional): List of indices indicating which inputs are used as inputs to the decoder. Defaults to [].
        input_output_idx (list, optional): List of indices indicating which inputs are used as outputs from the encoder. Defaults to [].
        encoder_output (torch.Tensor, optional): Output tensor from the encoder. Shape: (num_samples, input_len, encoder_output_size). Defaults to None.
        target (torch.Tensor, optional): Target tensor. Shape: (num_samples, output_len, output_size). Defaults to None.
        output_window_idx (list, optional): List of indices indicating which outputs are used for the output window. Defaults to None.

    Returns:
        torch.Tensor: Decoder output tensor of shape (num_samples, output_len, output_size).
        list: List of hidden states after the encoder.
    '''

    num_samples, input_len, input_size = input.shape

    encoder_steps = steps[:, :input_len] if steps is not None else None
    decoder_steps = steps[:, (input_len - 1):] if steps is not None else None

    encoder_output, encoder_hiddens = self.encoder(input=input,
                                                   steps=encoder_steps,
                                                   hiddens=hiddens,
                                                   input_mask=input_mask)

    hiddens = encoder_hiddens
    
    decoder_hiddens = [None for _ in range(self.decoder.num_inputs)]

    if self.enc2dec_hiddens_block is not None:
      # If the enc2dec_hiddens_block exists (decoder must contain rnn's)
      if encoder_hiddens is not None:
        # If there are rnn hiddens
        enc2dec_hiddens_input = []
        for i in range(self.encoder.num_inputs):
          enc2dec_hiddens_input.append(torch.cat(encoder_hiddens[i], -1).reshape(num_samples, -1) if self.encoder.base_type[i] == 'lstm' else encoder_hiddens[i].reshape(num_samples, -1))
        enc2dec_hiddens_input = torch.cat(enc2dec_hiddens_input, -1)
      else:
        enc2dec_hiddens_input = encoder_output

      enc2dec_hiddens_output = self.enc2dec_hiddens_block(enc2dec_hiddens_input)

      j = 0
      for i in range(self.decoder.num_inputs):
        if self.decoder.base_type[i] in ['gru', 'lstm', 'lru']:
          
          split_size = (1 + int(self.decoder.base_rnn_bidirectional[i]))*self.decoder.base_hidden_size[i]
          
          decoder_hiddens_i = enc2dec_hiddens_output[:, j:(j + self.decoder_hidden_size[i])].split(split_size, 1)

          if self.decoder.base_type[i] == 'lstm':
            decoder_hiddens[i] = [h.reshape(-1, num_samples, self.decoder.base_hidden_size[i]) for h in decoder_hiddens_i]
          else:
            decoder_hiddens[i] = decoder_hiddens_i[0].reshape(-1, num_samples, self.decoder.base_hidden_size[i])          
          j += self.decoder_hidden_size[i]
    else:
      decoder_hiddens = None
    
    # max_output_len = np.max([base.seq_len for base in self.decoder.seq_base])

    if self.enc_out_as_dec_in:
      decoder_input = encoder_output
      encoder_output = None
    else:
      decoder_input = self.enc2dec_input_block(input[:, -1:]) if self.enc2dec_input_block is not None else input[:, -1:]      
      decoder_input = torch.nn.functional.pad(decoder_input, (0, 0, 0, self.decoder.max_output_len - 1), "constant", 0)

    decoder_output, _ = self.decoder(input = decoder_input.clone(),
                                     steps = decoder_steps,
                                     hiddens = decoder_hiddens,
                                     target = target,
                                     output_window_idx = output_window_idx,
                                     output_mask = output_mask,
                                     output_input_idx = output_input_idx,
                                     input_output_idx = input_output_idx,
                                     encoder_output = encoder_output)
    
    return decoder_output, hiddens
