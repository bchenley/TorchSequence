import torch
import numpy as np

from ts_src import HiddenLayer as HiddenLayer

class Seq2SeqModel(torch.nn.Module):
    def __init__(self,
                 encoder, decoder,
                 learn_decoder_input=False, learn_decoder_hiddens=False,
                 enc2dec_bias=True, enc2dec_hiddens_bias=True,
                 enc2dec_dropout_p=0., enc2dec_hiddens_dropout_p=0.,
                 enc_out_as_dec_in=False,
                 device='cpu', dtype=torch.float32):
        """
        Initializes the Seq2SeqModel instance.

        Args:
            encoder (torch.nn.Module): The encoder component of the Seq2Seq model.
            decoder (torch.nn.Module): The decoder component of the Seq2Seq model.
            learn_decoder_input (bool): Whether to learn decoder input.
            learn_decoder_hiddens (bool): Whether to learn decoder hidden states.
            enc2dec_bias (bool): Whether to use bias in the enc2dec_input_block.
            enc2dec_hiddens_bias (bool): Whether to use bias in the enc2dec_hiddens_block.
            enc2dec_dropout_p (float): Dropout probability for the enc2dec_input_block.
            enc2dec_hiddens_dropout_p (float): Dropout probability for the enc2dec_hiddens_block.
            enc_out_as_dec_in (bool): Whether to use encoder output as decoder input.
            device (str): The device to be used for computations.
            dtype (torch.dtype): The data type to be used for computations.
        """
        super(Seq2SeqModel, self).__init__()

        # Save the input arguments as instance attributes.
        self.encoder = encoder
        self.decoder = decoder
        self.learn_decoder_input = learn_decoder_input
        self.learn_decoder_hiddens = learn_decoder_hiddens
        self.enc2dec_bias = enc2dec_bias
        self.enc2dec_hiddens_bias = enc2dec_hiddens_bias
        self.enc2dec_dropout_p = enc2dec_dropout_p
        self.enc2dec_hiddens_dropout_p = enc2dec_hiddens_dropout_p
        self.enc_out_as_dec_in = enc_out_as_dec_in
        self.device = device
        self.dtype = dtype

        # Get attributes from the encoder and decoder.
        self.num_inputs, self.num_outputs = self.encoder.num_inputs, self.decoder.num_outputs
        self.input_size, self.output_size = self.encoder.input_size, self.decoder.output_size
        self.base_type = self.encoder.base_type

        self.enc2dec_input_block = None

        # If learning decoder input, create an input block to map encoder output to decoder input.
        if self.learn_decoder_input:
            self.enc2dec_input_block = HiddenLayer(in_features=sum(self.encoder.input_size),
                                                   out_features=sum(self.decoder.input_size),
                                                   bias=self.enc2dec_bias,
                                                   activation='identity',
                                                   dropout_p=self.enc2dec_dropout_p,
                                                   device=self.device,
                                                   dtype=self.dtype)

        self.enc2dec_hiddens_block = None
        # If learning decoder hidden states, create a block to map encoder output to decoder hidden states.
        if self.learn_decoder_hiddens:
            if any(type_ in ['gru', 'lstm', 'lru'] for type_ in self.encoder.base_type):
                total_encoder_hidden_size = 0
                for i in range(self.encoder.num_inputs):
                    if self.encoder.base_type[i] in ['lstm', 'gru']:
                        total_encoder_hidden_size += (
                                    1 + int(self.encoder.base_rnn_bidirectional[i])) * self.encoder.base_hidden_size[i]
                    elif self.encoder.base_type[i] == 'lru':
                        total_encoder_hidden_size += self.encoder.base_num_filterbanks[i] * self.encoder.base_hidden_size[
                            i]

            else:
                total_encoder_hidden_size = sum(self.encoder.output_size)

            self.decoder_hidden_size = []
            for i in range(self.decoder.num_inputs):
                self.decoder_hidden_size.append(0)
                if self.decoder.base_type[i] in ['lstm', 'gru']:
                    self.decoder_hidden_size[i] = (
                                1 + int(self.decoder.base_rnn_bidirectional[i])) * self.decoder.base_num_layers[i] * \
                                                  self.decoder.base_hidden_size[i]
                elif self.decoder.base_type[i] == 'lru':
                    self.decoder_hidden_size[i] = self.decoder.base_num_filterbanks[i] * self.decoder.base_hidden_size[
                        i]

            self.enc2dec_hiddens_block = HiddenLayer(in_features=total_encoder_hidden_size,
                                                     out_features=sum(self.decoder_hidden_size),
                                                     bias=self.enc2dec_hiddens_bias,
                                                     activation='identity',
                                                     dropout_p=self.enc2dec_hiddens_dropout_p,
                                                     device=self.device,
                                                     dtype=self.dtype)

    def forward(self,
                input,
                steps=None,
                hiddens=None,
                input_mask=None, output_mask=None,
                output_input_idx=[], input_output_idx=[],
                encoder_output=None,
                target=None,
                input_window_idx=None,
                output_window_idx=None):
        """
        Performs the forward pass of the Seq2SeqModel.

        Args:
            input (torch.Tensor): The input data tensor.
            steps (torch.Tensor): The time steps for each sample in the input data.
            hiddens (list): The initial hidden states for the encoder.
            input_mask (torch.Tensor): The input mask for masking specific positions in the input.
            output_mask (torch.Tensor): The output mask for masking specific positions in the output.
            output_input_idx (list): List of indices for output-input masking.
            input_output_idx (list): List of indices for input-output masking.
            encoder_output (torch.Tensor): The output of the encoder.
            target (torch.Tensor): The target data tensor.
            input_window_idx (torch.Tensor): The window indices for input.
            output_window_idx (torch.Tensor): The window indices for output.

        Returns:
            torch.Tensor: The decoder output.
            list: The hidden states of the encoder.
        """
        num_samples, input_len, input_size = input.shape

        # Split steps into encoder steps and decoder steps.
        encoder_steps = steps[:, :input_len] if steps is not None else None
        decoder_steps = steps[:, (input_len - 1):] if steps is not None else None

        # Pass the input through the encoder to obtain encoder output and hidden states.
        encoder_output, encoder_hiddens = self.encoder(input=input,
                                                       steps=encoder_steps,
                                                       hiddens=hiddens,
                                                       input_mask=input_mask)

        hiddens = encoder_hiddens

        decoder_hiddens = [None for _ in range(self.decoder.num_inputs)]

        # If learning decoder hidden states, map encoder output to decoder hidden states.
        if self.enc2dec_hiddens_block is not None:
            if any(type_ in ['gru', 'lstm', 'lru'] for type_ in self.encoder.base_type):
                enc2dec_hiddens_input = []
                for i in range(self.encoder.num_inputs):
                    if self.encoder.base_type[i] == 'lstm':
                        enc2dec_hiddens_input.append(encoder_hiddens[i][0][-1:].reshape(num_samples, -1))
                    elif self.encoder.base_type[i] == 'gru':
                        enc2dec_hiddens_input.append(encoder_hiddens[i][-1:].reshape(num_samples, -1))
                    elif self.encoder.base_type[i] == 'lru':
                        enc2dec_hiddens_input.append(encoder_hiddens[i][0].reshape(num_samples, -1))

                enc2dec_hiddens_input = torch.cat(enc2dec_hiddens_input, -1)
            else:
                enc2dec_hiddens_input = encoder_output.reshape(num_samples, -1)

            enc2dec_hiddens_output = self.enc2dec_hiddens_block(enc2dec_hiddens_input)

            j = 0
            for i in range(self.decoder.num_inputs):
                if self.decoder.base_type[i] == 'lstm':
                    decoder_hiddens[i] = [
                        enc2dec_hiddens_output[:, j:(j + self.decoder_hidden_size[i])].reshape(-1, num_samples, self.decoder.base_hidden_size[i])]
                    decoder_hiddens[i].append(torch.zeros_like(decoder_hiddens[i][0]))
                if self.decoder.base_type[i] == 'gru':
                    decoder_hiddens[i] = enc2dec_hiddens_output[:, j:(j + self.decoder_hidden_size[i])].reshape(-1, num_samples, self.decoder.base_hidden_size[i])
                if self.decoder.base_type[i] == 'lru':
                    decoder_hiddens[i] = enc2dec_hiddens_output[:, j:(j + self.decoder_hidden_size[i])].reshape(-1, num_samples, self.decoder.base_hidden_size[i])

                j += self.decoder_hidden_size[i]

        else:
            decoder_hiddens = encoder_hiddens

        # Prepare decoder input based on the enc_out_as_dec_in flag.
        if self.enc_out_as_dec_in:
            decoder_input = encoder_output
        else:
            decoder_input = self.enc2dec_input_block(input[:, -1:]) if self.enc2dec_input_block is not None else input[:,-1:]
            decoder_input = torch.nn.functional.pad(decoder_input, (0, 0, 0, self.decoder.max_output_len - 1), "constant", 0)

        # Pass decoder input through the decoder to get the decoder output.
        decoder_output, _ = self.decoder(input=decoder_input.clone(),
                                         steps=decoder_steps,
                                         hiddens=decoder_hiddens,
                                         target=target,
                                         output_window_idx=output_window_idx,
                                         output_mask=output_mask,
                                         output_input_idx=output_input_idx,
                                         input_output_idx=input_output_idx,
                                         encoder_output=encoder_output)

        return decoder_output, hiddens
