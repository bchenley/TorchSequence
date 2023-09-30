import torch
from ts_src.Polynomial import Polynomial
from ts_src.Sigmoid import Sigmoid

class HiddenLayer(torch.nn.Module):
    """
    A configurable hidden layer for neural networks.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include bias terms.
        activation (str): Activation function for the layer.
        weight_to_ones (bool): If True, initializes weights to 1.
        weight_reg (list): Regularization parameters for weights: [lambda, p].
        weight_norm (int): Norm for weight normalization.
        degree (int): Degree for polynomial activation.
        coef_init (float): Initial coefficient value for polynomial activation.
        coef_train (bool): Whether to train polynomial coefficients.
        coef_reg (list): Regularization parameters for coefficients: [lambda, p].
        zero_order (bool): Whether to include zero-order term in polynomial activation.
        softmax_dim (int): Dimension for softmax activation.
        dropout_p (float): Dropout probability.
        sigmoid_slope_init (float): Initial value for sigmoid slope.
        sigmoid_slope_train (bool): Whether to train sigmoid slope.
        sigmoid_slope_reg (list): Regularization parameters for sigmoid slope: [lambda, p].
        sigmoid_shift_init (float): Initial value for sigmoid shift.
        sigmoid_shift_train (bool): Whether to train sigmoid shift.
        sigmoid_shift_reg (list): Regularization parameters for sigmoid shift: [lambda, p].
        sigmoid_bias (bool): Whether to include bias in sigmoid activation.
        norm_type (str): Type of normalization ('batch', 'layer', None).
        affine_norm (bool): Whether to use affine transformation in normalization.
        device (str): Device for computation ('cpu' or 'cuda').
        dtype (torch.dtype): Data type for tensors.
        F (torch.nn.Sequential): The core function of the hidden layer.
        dropout (torch.nn.Dropout): Dropout layer.
        norm_layer (torch.nn.Module): Normalization layer.
        group_mask (torch.Tensor): A mask used to apply weight sparsity.

    Args:
        ... [Same as original]

    Methods:
        forward(input): Forward pass through the layer.
        constrain(): Apply weight normalization to constrain weights.
        penalize(): Compute regularization loss for weights and coefficients.
    """

    def __init__(self, in_features, out_features=None, 
                 bias=True, activation='identity', groups = 1,
                 weight_to_ones=False, weight_reg=[0.001, 1], weight_norm=2, 
                 degree=1, coef_init=None, coef_train=True,
                 coef_reg=[0.001, 1], zero_order=False, softmax_dim=-1, dropout_p=0.0,
                 sigmoid_slope_init=None, sigmoid_slope_train=True, sigmoid_slope_reg=[0.001, 1],
                 sigmoid_shift_init=None, sigmoid_shift_train=True, sigmoid_shift_reg=[0.001, 1],
                 sigmoid_bias=True,
                 norm_type=None, affine_norm=False,
                 device='cpu', dtype=torch.float32):

        # Call the superclass constructor
        super(HiddenLayer, self).__init__()

        # Store all arguments as attributes
        locals_ = locals().copy()
        for arg in locals_:
            if arg != 'self':
                setattr(self, arg, locals_[arg])

        # Handle default output features
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

                if self.weight_to_ones or (self.in_features == 1):
                    f1.weight.data.fill_(1.0)
                    f1.weight.requires_grad = False

        # Handle activation functions
        if self.activation == 'identity':
            f2 = torch.nn.Identity()
        elif activation == 'polynomial':
            f2 = Polynomial(in_features=self.out_features, degree=self.degree, coef_init=self.coef_init,
                            coef_train=self.coef_train, coef_reg=self.coef_reg, zero_order=self.zero_order,
                            device=self.device, dtype=self.dtype)
            
        elif self.activation == 'tanh':
            f2 = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            f2 = Sigmoid(in_features = self.out_features, 
                         slope_init=self.sigmoid_slope_init, slope_train=self.sigmoid_slope_train, slope_reg=self.sigmoid_slope_reg,
                         shift_init=self.sigmoid_shift_init, shift_train=self.sigmoid_shift_train, shift_reg=self.sigmoid_shift_reg,
                         bias=self.sigmoid_bias,
                         device=self.device, dtype=self.dtype)            
        elif self.activation == 'softmax':
            f2 = torch.nn.Softmax(dim=self.softmax_dim)
        elif self.activation == 'relu':
            f2 = torch.nn.ReLU()
        else:
            raise ValueError(f"activation ({self.activation}) must be 'identity', 'polynomial', 'tanh', 'sigmoid', or 'relu'.")

        # Create the layer as a sequential composition of f1 and f2
        self.F = torch.nn.Sequential(f1, f2)

        # Add dropout
        self.dropout = torch.nn.Dropout(self.dropout_p)

        # Add normalization layer based on norm_type
        if self.norm_type == 'batch':            
            self.norm_layer = torch.nn.BatchNorm1d(out_features, 
                                                   affine=self.affine_norm,
                                                   device=self.device, dtype=self.dtype)
        elif self.norm_type == 'layer':
            self.norm_layer = torch.nn.LayerNorm(out_features, 
                                                 elementwise_affine=self.affine_norm,
                                                 device=self.device, dtype=self.dtype)
        else:
            self.norm_layer = torch.nn.Identity()

        if (self.groups > self.in_features) | (self.groups > self.out_features):
          raise ValueError(f"Number of groups ({self.groups}) cannot be larger than the number of input features ({self.in_features}) or output features ({self.out_features}).")

        # Calculate the number of full groups and the size of the last group
        row_size = self.out_features // self.groups
        col_size = self.in_features // self.groups
        
        # Determine group sizes
        row_sizes = [row_size] * self.groups
        col_sizes = [col_size] * self.groups

        if sum(row_sizes) < self.out_features: row_sizes[-1] += self.out_features - sum(row_sizes)
        if sum(col_sizes) < self.in_features: col_sizes[-1] += self.in_features - sum(col_sizes)
        
        # Generate the mask based on group sizes
        self.group_mask = torch.zeros_like(self.F[0].weight,
                                          requires_grad = False)       
        
        i,j = 0,0
        for row_size, col_size in zip(row_sizes, col_sizes):        
          self.group_mask[i:(i+row_size), j:(j+col_size)] = 1.          
          j += col_size
          i += row_size

    def forward(self, input):
        """
        Forward pass through the hidden layer.

        Args:
            input (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        self.F[0].weight.data *= self.group_mask
        
        output = self.dropout(self.F(input))

        if self.norm_type == 'batch':
            output = self.norm_layer(output.permute(0, 2, 1)).permute(0, 2, 1)
        if self.norm_type == 'layer':
            output = self.norm_layer(output)
        
        return output

    def constrain(self):
        """
        Apply weight normalization to constrain weights.
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                param = torch.nn.functional.normalize(param, p=self.weight_norm, dim=1).contiguous()

    def penalize(self):
        """
        Compute regularization loss for weights and coefficients.

        Returns:
            torch.Tensor: Regularization loss.
        """
        loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                loss += self.weight_reg[0] * torch.norm(param, p=self.weight_reg[1]) * int(param.requires_grad)
            elif 'coef' in name:
                loss += self.coef_reg[0] * torch.norm(param, p=self.coef_reg[1]) * int(param.requires_grad)

        return loss
