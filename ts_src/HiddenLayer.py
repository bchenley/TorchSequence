import torch

from ts_src.Polynomial import Polynomial
from ts_src.Sigmoid import Sigmoid

class HiddenLayer(torch.nn.Module):

    def __init__(self, in_features, out_features=None, 
                 bias=True, activation='identity',
                 weights_to_1=False, weight_reg=[0.001, 1], weight_norm=2, 
                 degree=1, coef_init=None, coef_train=True,
                 coef_reg=[0.001, 1], zero_order=False, softmax_dim=-1, dropout_p=0.0,
                 sigmoid_slope_init = None, sigmoid_slope_train = True, sigmoid_slope_reg=[0.001, 1],
                 sigmoid_shift_init = None, sigmoid_shift_train = True, sigmoid_shift_reg=[0.001, 1],
                 sigmoid_bias = True,
                 norm_type = None, affine_norm = False,
                 device='cpu', dtype=torch.float32):

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
            f2 = Sigmoid(in_features, 
                         slope_init = sigmoid_slope_init, slope_train = sigmoid_slope_train, slope_reg=sigmoid_slope_reg,
                         shift_init = sigmoid_shift_init, shift_train = sigmoid_shift_train, shift_reg=sigmoid_shift_reg,
                         bias = sigmoid_bias,
                         device=self.device, dtype=self.dtype)            
        elif self.activation == 'softmax':
            f2 = torch.nn.Softmax(dim=self.softmax_dim)
        elif self.activation == 'relu':
            f2 = torch.nn.ReLU()
        else:
            raise ValueError(f"activation ({self.activation}) must be 'identity', 'polynomial', 'tanh', 'sigmoid', or 'relu'.")

        self.F = torch.nn.Sequential(f1, f2)

        self.dropout = torch.nn.Dropout(self.dropout_p)

        if self.norm_type == 'batch':            
            self.norm_layer = torch.nn.BatchNorm1d(out_features, 
                                                   affine = self.affine_norm,
                                                   device = self.device, dtype = self.dtype)
        elif self.norm_type == 'layer':
            self.norm_layer = torch.nn.LayerNorm(out_features, 
                                                 elementwise_affine = self.affine_norm,
                                                 device = self.device, dtype = self.dtype)
        else:
            self.norm_layer = torch.nn.Identity()

    def forward(self, input):

        output = self.dropout(self.F(input))

        if self.norm_type == 'batch':
            output = self.norm_layer(output.permute(0, 2, 1)).permute(0, 2, 1)
        if self.norm_type == 'layer':
            output = self.norm_layer(output)
        
        return output

    def constrain(self):

        for name, param in self.named_parameters():
            if 'weight' in name:
                param = torch.nn.functional.normalize(param, p=self.weight_norm, dim=1).contiguous()

    def penalize(self):

        loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                loss += self.weight_reg[0] * torch.norm(param, p=self.weight_reg[1]) * int(param.requires_grad)
            elif 'coef' in name:
                loss += self.coef_reg[0] * torch.norm(param, p=self.coef_reg[1]) * int(param.requires_grad)

        return loss
