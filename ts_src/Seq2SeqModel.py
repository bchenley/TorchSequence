import torch
import numpy as np

from ts_src import HiddenLayer as HiddenLayer

class Seq2SeqModel(torch.nn.Module):
    def __init__(self,
                 encoder, decoder, enc2dec_hiddens_type=None,
                 learn_decoder_input=False,
                 enc2dec_bias=True, enc2dec_hiddens_bias=True,
                 enc2dec_dropout_p=0., enc2dec_hiddens_dropout_p=0.,
                 enc_out_as_dec_in=False,
                 device='cpu', dtype=torch.float32):

      """
      Initialize the Seq2SeqModel.

      Args:
        encoder (Encoder): The encoder object.
        decoder (Decoder): The decoder object.
        enc2dec_hiddens_type (str, optional): Type of enc2dec hidden state computation ('learn' or 'identity'). Default is None.
        learn_decoder_input (bool, optional): Whether to learn decoder input mapping. Default is False.
        enc2dec_bias (bool, optional): Whether to use bias in enc2dec input block. Default is True.
        enc2dec_hiddens_bias (bool, optional): Whether to use bias in enc2dec hiddens block. Default is True.
        enc2dec_dropout_p (float, optional): Dropout probability for enc2dec input block. Default is 0.
        enc2dec_hiddens_dropout_p (float, optional): Dropout probability for enc2dec hiddens block. Default is 0.
        enc_out_as_dec_in (bool, optional): Whether to use encoder output as decoder input. Default is False.
        device (str, optional): Device for computation ('cpu' or 'cuda'). Default is 'cpu'.
        dtype (torch.dtype, optional): Data type for computation. Default is torch.float32.
      """

      # Call the constructor of the base class
      super(Seq2SeqModel, self).__init__()

      # Create a dictionary of local variables
      locals_ = locals().copy()

      # Loop through the local variables
      for arg in locals_:
        if arg != 'self':
          # Set the attribute on the instance using the variable name and its value
          setattr(self, arg, locals_[arg])

      # Set attributes based on encoder and decoder attributes
      self.dt = self.encoder.dt
      self.total_input_size, self.total_output_size = self.encoder.total_input_size, self.decoder.total_output_size
      self.max_input_len, self.max_output_len = self.encoder.max_input_len, self.decoder.max_output_len

      # Set the number of inputs and outputs based on encoder and decoder attributes
      self.num_inputs, self.num_outputs = self.encoder.num_inputs, self.decoder.num_outputs
      self.input_size, self.output_size = self.encoder.input_size, self.decoder.output_size
      self.base_type = self.encoder.base_type

      # Initialize enc2dec input block
      self.enc2dec_input_block = None
      if self.learn_decoder_input:
        self.enc2dec_input_block = HiddenLayer(in_features = sum(self.encoder.input_size),
                                               out_features = sum(self.decoder.input_size),
                                               bias = self.enc2dec_bias,
                                               activation = 'identity',
                                               dropout_p = self.enc2dec_dropout_p,
                                               device = self.device,
                                               dtype = self.dtype)

      # Initialize enc2dec hiddens block
      self.enc2dec_hiddens_block = None
      if self.enc2dec_hiddens_type == 'learn':
        # Calculate total encoder hidden size based on encoder attributes
        if any(type_ in ['gru', 'lstm', 'lru'] for type_ in self.encoder.base_type):
          total_encoder_hidden_size = 0
          for i in range(self.encoder.num_inputs):
            if self.encoder.base_type[i] in ['lstm', 'gru']:
              total_encoder_hidden_size += (1 + int(self.encoder.base_rnn_bidirectional[i])) * self.encoder.base_hidden_size[i]
            elif self.encoder.base_type[i] == 'lru':
              total_encoder_hidden_size += self.encoder.base_num_filterbanks[i] * self.encoder.base_hidden_size[i]
        else:
          total_encoder_hidden_size = sum(self.encoder.output_size)

        # Calculate decoder hidden sizes based on decoder attributes
        self.decoder_hidden_size = []
        for i in range(self.decoder.num_inputs):
          self.decoder_hidden_size.append(0)
          if self.decoder.base_type[i] in ['lstm', 'gru']:
            self.decoder_hidden_size[i] = (1 + int(self.decoder.base_rnn_bidirectional[i])) * self.decoder.base_num_layers[i] * self.decoder.base_hidden_size[i]
          elif self.decoder.base_type[i] == 'lru':
            self.decoder_hidden_size[i] = self.decoder.base_num_filterbanks[i] * self.decoder.base_hidden_size[i]

        # Initialize enc2dec hiddens block
        self.enc2dec_hiddens_block = HiddenLayer(in_features = total_encoder_hidden_size,
                                                 out_features = sum(self.decoder_hidden_size),
                                                 bias = self.enc2dec_hiddens_bias,
                                                 activation = 'identity',
                                                 dropout_p = self.enc2dec_hiddens_dropout_p,
                                                 device = self.device,
                                                 dtype = self.dtype)

    def forward(self,
                input,
                steps = None,
                hiddens = None,
                input_mask = None, output_mask = None,
                input_window_idx = None, output_window_idx = None,
                output_input_idx = [], input_output_idx = [],
                encoder_output = None,
                target = None):

      """
      Forward pass of the Seq2SeqModel.

      Args:
          input (Tensor): Input tensor of shape (batch_size, input_len, input_size).
          steps (Tensor): Time step tensor of shape (batch_size, input_len).
          hiddens (list of Tensors): Hidden states from previous step.
          input_mask (Tensor): Input mask tensor of shape (batch_size, input_len).
          output_mask (Tensor): Output mask tensor of shape (batch_size, output_len).
          input_window_idx (list of Tensors): Indices for input windows. Default is None.
          output_window_idx (list of Tensors): Indices for output windows. Default is None.
          output_input_idx (list): List of indices for output inputs. Default is an empty list.
          input_output_idx (list): List of indices for input outputs. Default is an empty list.
          encoder_output (Tensor): Output from the encoder.
          target (Tensor): Target tensor of shape (batch_size, output_len, output_size).

      Returns:
          decoder_output (Tensor): Decoder output tensor.
          hiddens (Tensor): Hidden states tensor.
      """

      # Get the number of samples, input length, and input size
      num_samples, input_len, input_size = input.shape

      # Prepare input window indices if not provided
      input_window_idx = [torch.arange(input_len).to(device=self.device, dtype=torch.long)
                          for _ in range(self.num_inputs)] if input_window_idx is None else input_window_idx
      output_window_idx = [torch.arange(input_len).to(device=self.device, dtype=torch.long)
                           for _ in range(self.num_outputs)] if output_window_idx is None else output_window_idx

      unique_input_window_idx = torch.cat(input_window_idx).unique()
      unique_output_window_idx = torch.cat(output_window_idx).unique()

      # Get encoder and decoder steps
      encoder_steps = steps[:, unique_input_window_idx] if steps is not None else None
      decoder_steps = steps[:, unique_output_window_idx] if steps is not None else None

      # Perform encoder forward pass
      encoder_output, encoder_hiddens = self.encoder(input = input,
                                                     steps = encoder_steps,
                                                     hiddens = hiddens,
                                                     input_mask = input_mask)
      
      # hiddens = encoder_hiddens.copy()
      
      # Compute decoder hidden states based on enc2dec hiddens type
      if self.enc2dec_hiddens_type is None:
          decoder_hiddens = None
      elif self.enc2dec_hiddens_type == 'learn':
          # Initialize decoder hidden states as None
          decoder_hiddens = [None for _ in range(self.decoder.num_inputs)]
          # Compute enc2dec hiddens input based on encoder hidden states
          enc2dec_hiddens_input = []
          for i in range(self.encoder.num_inputs):
              if self.encoder.base_type[i] == 'lstm':
                enc2dec_hiddens_input.append(encoder_hiddens[i][0][-1:].reshape(num_samples, -1))
              elif self.encoder.base_type[i] == 'gru':
                enc2dec_hiddens_input.append(encoder_hiddens[i][-1:].reshape(num_samples, -1))
              elif self.encoder.base_type[i] == 'lru':
                  enc2dec_hiddens_input.append(encoder_hiddens[i][0].reshape(num_samples, -1))
          enc2dec_hiddens_input = torch.cat(enc2dec_hiddens_input, -1)

          # Compute enc2dec hiddens output using enc2dec hiddens block
          enc2dec_hiddens_output = self.enc2dec_hiddens_block(enc2dec_hiddens_input)

          # Initialize index counter
          j = 0
          for i in range(self.decoder.num_inputs):
            if self.decoder.base_type[i] == 'lstm':
              decoder_hiddens[i] = (enc2dec_hiddens_output[:, j:(j + self.decoder_hidden_size[i])].reshape(-1, num_samples, self.decoder.base_hidden_size[i]),)
              decoder_hiddens[i] += (torch.zeros_like(decoder_hiddens[i][0]),)
            if self.decoder.base_type[i] == 'gru':
              decoder_hiddens[i] = enc2dec_hiddens_output[:, j:(j + self.decoder_hidden_size[i])].reshape(-1, num_samples, self.decoder.base_hidden_size[i])
            if self.decoder.base_type[i] == 'lru':
              decoder_hiddens[i] = enc2dec_hiddens_output[:, j:(j + self.decoder_hidden_size[i])].reshape(-1, num_samples, self.decoder.base_hidden_size[i])
            j += self.decoder_hidden_size[i]
      
      elif self.enc2dec_hiddens_type == 'identity':
        # Use encoder hidden states as decoder hidden states
        decoder_hiddens = encoder_hiddens[-self.decoder.num_outputs:]
        for i in range(self.decoder.num_inputs):
          if self.decoder.base_type[i] == 'lstm':
            decoder_hiddens[i] = (decoder_hiddens[i][0], torch.zeros_like(decoder_hiddens[i][1]))

      if self.enc_out_as_dec_in:
        decoder_input = encoder_output.clone()
        decoder_steps = None # steps
      else:
        input_slice = input.clone()[:, -1:]
        if len(output_input_idx) > 0:
          input_slice = input_slice[:, :, output_input_idx]          
        if self.enc2dec_input_block is not None:
          decoder_input = self.enc2dec_input_block(input_slice)
        else:
          decoder_input = input_slice
        
        decoder_input = decoder_input.reshape(num_samples, 1, self.decoder.total_output_size)
        
        # Pad decoder input
        decoder_input = torch.nn.functional.pad(decoder_input, (0, 0, 0, self.decoder.max_output_len - 1), "constant", 0)
    
      # Perform decoder forward pass
      decoder_output, decoder_hiddens = self.decoder(input = decoder_input,
                                                     steps = decoder_steps,
                                                     hiddens = decoder_hiddens,
                                                     target = target,
                                                     output_mask = output_mask,
                                                     output_input_idx = output_input_idx,
                                                     input_output_idx = input_output_idx,
                                                     encoder_output = encoder_output)

      # if target is not None:
      #   plt.figure(num=2)
      #   plt.plot(encoder_steps[0].cpu(), input[0].cpu(), '-*b', label = 'input')
      #   plt.plot(decoder_steps[0].cpu(), target[0].cpu(), '-*k', label = 'target')
      #   plt.plot(decoder_steps[0].cpu(), decoder_input[0].detach().cpu(), '-*g', label = 'decoder input')
      #   plt.plot(decoder_steps[0].cpu(), decoder_output[0].detach().cpu(), '-*r', label = 'decoder output')
      #   plt.legend()

      return decoder_output, hiddens

    def constrain(self):
      """
      Apply constraints to the encoder and decoder components of the model.
      """
      # Apply constraints to the encoder
      self.encoder.constrain()

      # Apply constraints to the decoder
      self.decoder.constrain()

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

      # Get the number of samples, input length, and input size
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

      This method performs forecasting using the model. It takes input data, optionally steps data,
      and other parameters to generate forecast predictions. It returns the forecasted results and associated time steps.

      Args:
          input (Tensor): Input data tensor.
          steps (Tensor, optional): Steps data tensor. Default is None.
          hiddens (list of Tensors, optional): Initial hidden state tensors. Default is None.
          num_forecast_steps (int, optional): Number of forecast steps. Default is 1.
          encoder_output (Tensor, optional): Encoder output tensor. Default is None.
          input_window_idx (list of Tensors, optional): Indices for input windows. Default is None.
          output_window_idx (list of Tensors, optional): Indices for output windows. Default is None.
          input_mask (Tensor, optional): Input mask tensor. Default is None.
          output_mask (Tensor, optional): Output mask tensor. Default is None.
          output_input_idx (list, optional): Indices for output input. Default is an empty list.
          input_output_idx (list, optional): Indices for input output. Default is an empty list.
          output_transforms (list of Transform objects, optional): Output transforms for forecast. Default is None.

      Returns:
          forecast (Tensor): Forecasted results tensor.
          forecast_time (Tensor): Associated time steps tensor.
      """

      # Clone input and steps if provided
      input = input.clone()
      steps = steps.clone() if steps is not None else None

      with torch.no_grad():
          num_samples, input_len, input_size = input.shape

          # Initialize empty forecast tensor
          forecast = torch.empty((num_samples, 0, self.total_output_size)).to(device=self.device,
                                                                          dtype=self.dtype)

          # Initialize empty forecast steps tensor if steps are provided
          if steps is not None:
              forecast_steps = torch.empty((num_samples, 0)).to(steps)
          else:
              forecast_steps = None

          # Calculate the forecast length based on input and output window indices
          if (input_window_idx is not None) & (output_window_idx is not None):
              max_input_window_idx = np.max([idx.max().cpu() for idx in input_window_idx])
              max_output_window_idx = np.max([idx.max().cpu() for idx in output_window_idx])
              forecast_len = np.max([1, max_output_window_idx - max_input_window_idx])
          else:
              forecast_len = 1

          # Generate initial prediction and concatenate it to the forecast tensor
          prediction, hiddens = self.forward(input=input,
                                            steps=steps,
                                            hiddens=hiddens,
                                            input_window_idx=input_window_idx,
                                            output_window_idx=output_window_idx,
                                            encoder_output=encoder_output,
                                            input_output_idx=input_output_idx,
                                            output_input_idx=output_input_idx)
          forecast = torch.cat((forecast, prediction[:, -forecast_len:]), 1)

          # Update forecast steps tensor if steps are provided
          if steps is not None:
            forecast_steps = torch.cat((forecast_steps, steps[:, -forecast_len:]), 1)
            steps += forecast_len

          # Generate forecasts for the remaining steps
          while forecast.shape[1] < (forecast_len + num_forecast_steps):
            input_ar = torch.zeros((num_samples, forecast_len, input_size)).to(input)

            # Update input_ar with the forecasted values if needed
            if (len(input_output_idx) > 0) & (len(output_input_idx) > 0):
                input_ar[..., output_input_idx] = forecast[:, -forecast_len:, input_output_idx]

            # Concatenate input_ar with the input tensor
            input = torch.cat((input[:, forecast_len:], input_ar), 1)

            # Generate prediction for the next forecast step and concatenate it to the forecast tensor
            prediction, hiddens = self(input=input,
                                        steps=steps,
                                        hiddens=hiddens,
                                        encoder_output=encoder_output,
                                        input_output_idx=input_output_idx,
                                        output_input_idx=output_input_idx)

            forecast = torch.cat((forecast, prediction[:, -forecast_len:]), 1)

            # Update forecast steps tensor if steps are provided
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

      # Extract forecasted results for the specified number of forecast steps
      forecast = forecast[:, forecast_len:][:, :num_forecast_steps]
      forecast_steps = forecast_steps[:, forecast_len:][:, :num_forecast_steps] if forecast_steps is not None else None

      # Compute forecast time steps
      forecast_time = forecast_steps * self.dt if forecast_steps is not None else None

      return forecast, forecast_time
