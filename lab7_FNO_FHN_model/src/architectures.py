"""
architectures.py

author: Edoardo Centofanti

some pytorch architectures for DeepONet and similia.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn

#########################################
# Utilities
#########################################

def activation(act_fun):
    act_fun = act_fun.lower()
    act_dict = {
        "relu"     : F.relu,
        "tanh"     : F.tanh,
        "gelu"     : F.gelu,
        "sigmoid"  : F.sigmoid,
        "sin"      : lambda x: torch.sin(2*torch.pi*x),
    }
    return act_dict[act_fun]
    
def initializer(initial):
    initial = initial.lower()
    initial_dict = {
        "glorot normal": torch.nn.init.xavier_normal_,
        "glorot uniform": torch.nn.init.xavier_uniform_,
        "he normal": torch.nn.init.kaiming_normal_,
        "he uniform": torch.nn.init.kaiming_uniform_,
        "zeros": torch.nn.init.zeros_,
    }
    return initial_dict[initial]

def get_optimizer(model,lr,schedulerName):
    # AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-4)
    # lr policy
    scheduler = None
    if schedulerName is not None:
        if schedulerName.lower() == "reduceonplateau":
            ### Find the reduce on plateau scheduler on the docs and complete the code here
            ### SCHEDULER CODE HERE
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 10)
        

        else:
            raise ValueError("This scheduler has not been implemented yet.")
    else:
        schedulerName = "None"

    return optimizer, schedulerName, scheduler

#########################################
# Adaptive Linear
#########################################
class AdaptiveLinear(nn.Linear):
    """Applies a linear transformation to the input data as follows
    :math:`y = naxA^T + b`.
    More details available in Jagtap, A. D. et al. Locally adaptive
    activation functions with slope recovery for deep and
    physics-informed neural networks, Proc. R. Soc. 2020.

    Parameters
    ----------
    in_features : int
        The size of each input sample
    out_features : int 
        The size of each output sample
    bias : bool, optional
        If set to ``False``, the layer will not learn an additive bias
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions
    """
    def __init__(self, in_features, out_features, bias=True, adaptive_rate=None, adaptive_rate_scaler=None):
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        if self.adaptive_rate:
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(self.in_features))
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0
            
    def forward(self, input):
        if self.adaptive_rate:
            return nn.functional.linear(self.adaptive_rate_scaler * self.A * input, self.weight, self.bias)
        return nn.functional.linear(input, self.weight, self.bias)

#########################################
# MLP
#########################################
class MLP(nn.Module):
    """ shallow neural network """
    def __init__(self, in_channels, out_channels, mid_channels, act_fun="ReLu", arc=None):
        super(MLP, self).__init__()
        if arc == "FNO":
            self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
            self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)
        else:
            self.mlp1 = nn.Linear(in_channels, mid_channels)
            self.mlp2 = nn.Linear(mid_channels, out_channels)
        self.activation = activation(act_fun)

    def forward(self, x):
        x = self.mlp1(x)       # affine transformation
        x = self.activation(x) # activation function
        x = self.mlp2(x)       # affine transformation
        return x

#########################################
# Loss functions
#########################################
class L2relLoss():
    def __init__(self):
        self.name = "L2_rel"

    def get_name(self):
        return self.name
        
    """ sum of relative L^2 error """        
    def rel(self, x, y):
        diff_norms = torch.norm(x - y, 2, 1)
        y_norms = torch.norm(y, 2, 1)    
        return torch.sum(diff_norms/y_norms)
    
    def __call__(self, x, y):
        return self.rel(x, y)

def get_loss(name):
    name = name.lower()
    if name == "l2":
        return L2relLoss()
    else:
        raise ValueError("This loss has not been implemented yet.")
    


#########################################
# FNN class
#########################################  
class FNN(nn.Module):
    def __init__(self, layer_sizes, activation_str, kernel_initializer,adapt_actfun=False):
        super().__init__()
        self.layers      = layer_sizes
        self.activation  = activation(activation_str)
        self.initializer = initializer(kernel_initializer)
        self.linears     = nn.ModuleList()
        self.adapt_rate  = None
        
        if adapt_actfun:
            self.adapt_rate = 0.1

        # Assembly the network
        for i in range(1,len(layer_sizes)):
            self.linears.append(
                AdaptiveLinear(layer_sizes[i-1],layer_sizes[i],adaptive_rate=self.adapt_rate)
            )
            # Initialize the parameters
            self.initializer(self.linears[-1].weight)
            # Initialize bias to zero
            initializer("zeros")(self.linears[-1].bias) 
    
    def forward(self,x):
        for linear in self.linears[:-1]:
            x = self.activation(linear(x))
        x = self.linears[-1](x)
        return x

#########################################
# FNN_LN class
#########################################         
class FNN_LN(nn.Module):
    def __init__(self, layers, activation_str, initialization_str, adapt_actfun=False):
        super().__init__()
        self.layers = layers # list with the number of neurons for each layer
        self.activation_str = activation_str
        self.initialization_str = initialization_str
        self.adapt_rate  = None
        
        if adapt_actfun:
            self.adapt_rate = 0.1
        # linear layers
        self.linears = nn.ModuleList(
            [ AdaptiveLinear(self.layers[i],self.layers[i+1],adaptive_rate=self.adapt_rate) 
              for i in range( len(self.layers) - 1 ) ])
        
        # batch normalization apllied in hidden layers
        self.layer_norm = nn.ModuleList(
            [ nn.LayerNorm(self.layers[i])
              for i in range(1, len(self.layers) - 2) ])

        self.linears.apply(self.param_initialization)
            
    #  Initialization for parameters
    def param_initialization(self, m):        
        if type(m) == nn.Linear:
            #### calculate gain 
            if self.activation_str == "tanh" or self.activation_str == "relu":
                gain = nn.init.calculate_gain(self.activation_str)
                a = 0
            elif self.activation_str == "leaky_relu":
                gain = nn.init.calculate_gain(self.activation_str, 0.01)
                a = 0.01
            else:
                gain = 1
                a = 0.01
            
            #### weights initialization
            if self.initialization_str == "xavier_uniform":
                torch.nn.init.xavier_uniform_(m.weight.data, gain = gain)
                
            elif self.initialization_str == "xavier_normal":
                torch.nn.init.xavier_normal_(m.weight.data, gain = gain)
                
            elif self.initialization_str == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(m.weight.data, 
                                               a = a, 
                                               nonlinearity = self.activation_str)
                
            elif self.initialization_str == "kaiming_normal":
                torch.nn.init.kaiming_normal_(m.weight.data, 
                                               a = a, 
                                               nonlinearity = self.activation_str)
            #### bias initialization
            torch.nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = activation(self.activation_str)(self.linears[0](x))
        for i in range(1, len(self.layers) - 2):
            x = activation(self.activation_str)(self.linears[i](self.layer_norm[i-1](x)))
        return self.linears[-1](x)