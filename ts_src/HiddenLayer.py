import torch
from ts_src.Polynomial import Polynomial

class HiddenLayer(torch.nn.Module):
    '''
    Hidden layer module with various activation functions and regularization options.

    Args:
        in_features (int): Number of input features.
        out_features (int or None): Number of output features. If None or 0, output features will be the same as input features.
        bias (bool): If True, adds a learnable bias to the output.
        activation (str): Activation function to use. Options: 'identity', 'polynomial', 'tanh', 'sigmoid', 'softmax', 'relu'.
        weight_reg (list): Regularization parameters for the weights. [Regularization weight, regularization exponent]
        weight_norm (int): Norm to be used for weight regularization.
        degree (int): Degree of the polynomial activation function.
        coef_init (torch.Tensor): Initial coefficients for the polynomial activation function. If None, coefficients are initialized randomly.
        coef_train (bool): Whether to train the coefficients.
        coef_reg (list): Regularization parameters for the coefficients. [Regularization weight, regularization exponent]
        zero_order (bool): Whether to include the zeroth-order term (constant) in the polynomial activation function.
        softmax_dim (int): Dimension along which to apply softmax activation.
        dropout_p (float): Dropout probability.
        batch_norm (bool): If True, applies batch normalization to the output of the hidden layer.
        affine_norm (bool): If True, allows batch normalization to learn affine parameters (scale and shift).
        device (str): Device to use for computation ('cpu' or 'cuda').
        dtype (torch.dtype): Data type of the model parameters.

    '''
    def __init__(self, in_features, out_features=None, 
                 bias=True, activation='identity',
                 weights_to_1=False, weight_reg=[0.001, 1], weight_norm=2, 
                 degree=1, coef_init=None, coef_train=True,
                 coef_reg=[0.001, 1], zero_order=False, softmax_dim=-1, dropout_p=0.0,
                 norm_type = None, affine_norm = False,
                 device='cpu', dtype=torch.float32):
        '''
        Constructor method for initializing the HiddenLayer module and its attributes.

        Args:
            in_features (int): Number of input features.
            out_features (int or None): Number of output features. If None or 0, output features will be the same as input features.
            bias (bool): If True, adds a learnable bias to the output.
            activation (str): Activation function to use. Options: 'identity', 'polynomial', 'tanh', 'sigmoid', 'softmax', 'relu'.
            weights_to_1 (bool): If True, set the weights to 1.0 for Linear or Bilinear layers with input and output features equal to 1.
            weight_reg (list): Regularization parameters for the weights. [Regularization weight, regularization exponent]
            weight_norm (int): Norm to be used for weight regularization.
            degree (int): Degree of the polynomial activation function.
            coef_init (torch.Tensor): Initial coefficients for the polynomial activation function. If None, coefficients are initialized randomly.
            coef_train (bool): Whether to train the coefficients.
            coef_reg (list): Regularization parameters for the coefficients. [Regularization weight, regularization exponent]
            zero_order (bool): Whether to include the zeroth-order term (constant) in the polynomial activation function.
            softmax_dim (int): Dimension along which to apply softmax activation.
            dropout_p (float): Dropout probability.
            norm_type (bool): If True, applies batch normalization to the output of the hidden layer.
            affine_norm (bool): If True, allows batch normalization to learn affine parameters (scale and shift).
            device (str): Device to use for computation ('cpu' or 'cuda').
            dtype (torch.dtype): Data type of the model parameters.
        '''
        super(HiddenLayer, self).__init__()

        locals_ = locals().copy()
        for arg in locals_:
            if arg != 'self':
                setattr(self, arg, locals_[arg])
                
        if self.out_features is None or self.out_features == 0:
            self.out_features = self.in_features
            f1 = torch.nn.Identity()
        else:
            if isinstance(self.in_features, list):  # bilinear (must be len = 2)
                class Bilinear(torch.nn.Module):
                    def __init__(self, in1_features=self.in_features[0], in2_features=self.in_features[1],
                                 out_features=self.out_features, bias=self.bias, device=self.device, dtype=self.dtype):
                        super(Bilinear, self).__init__()

                        self.F = torch.nn.Bilinear(in1_features, in2_features, out_features, bias)

                    def forward(self, input):
                        input1, input2 = input
                        return self.F(input1, input2)

                f1 = Bilinear()
            else:
                f1 = torch.nn.Linear(in_features=self.in_features, out_features=self.out_features,
                                     bias=self.bias, device=self.device, dtype=self.dtype)

                if self.weights_to_1: # (self.in_features == 1) & (self.out_features == 1):
                    f1.weight.data.fill_(1.0)
                    f1.weight.requires_grad = False

        if self.activation == 'identity':
            f2 = torch.nn.Identity()
        elif activation == 'polynomial':
            f2 = Polynomial(in_features=self.out_features, degree=self.degree, coef_init=self.coef_init,
                            coef_train=self.coef_train, coef_reg=self.coef_reg, zero_order=self.zero_order,
                            device=self.device, dtype=self.dtype)

        elif self.activation == 'tanh':
            f2 = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            f2 = torch.nn.Sigmoid()
        elif self.activation == 'softmax':
            f2 = torch.nn.Softmax(dim=self.softmax_dim)
        elif self.activation == 'relu':
            f2 = torch.nn.ReLU()
        else:
            raise ValueError(f"activation ({self.activation}) must be 'identity', 'polynomial', 'tanh', 'sigmoid', or 'relu'.")

        self.F = torch.nn.Sequential(f1, f2)

        self.dropout = torch.nn.Dropout(self.dropout_p)

        if self.norm_type == 'batch':            
            self.norm_layer = torch.nn.BatchNorm2d(out_features, 
                                                   affine = self.affine_norm,
                                                   device = self.device, dtype = self.dtype)
        elif self.norm_type == 'layer':
            self.norm_layer = torch.nn.LayerNorm(out_features, 
                                                 elementwise_affine = self.affine_norm,
                                                 device = self.device, dtype = self.dtype)
        else:
            self.norm_layer = torch.nn.Identity()

    def forward(self, input):
        '''
        Perform a forward pass through the hidden layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        '''
        y = self.norm_layer(self.dropout(self.F(input)))
        return y

    def constrain(self):
        '''
        Constrain the weights of the hidden layer.

        '''
        for name, param in self.named_parameters():
            if 'weight' in name:
                param = torch.nn.functional.normalize(param, p=self.weight_norm, dim=1).contiguous()

    def penalize(self):
        '''
        Compute the regularization loss for the hidden layer.

        Returns:
            torch.Tensor: Regularization loss.

        '''
        loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                loss += self.weight_reg[0] * torch.norm(param, p=self.weight_reg[1]) * int(param.requires_grad)
            elif 'coef' in name:
                loss += self.coef_reg[0] * torch.norm(param, p=self.coef_reg[1]) * int(param.requires_grad)

        return loss
