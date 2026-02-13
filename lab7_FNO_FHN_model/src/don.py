import torch
import torch.nn as nn
from .architectures import FNN, FNN_LN
from .architectures import activation

#########################################
# DeepONet class
#########################################  
class DeepONet(nn.Module):
    def __init__(self, layers, activation_str, kernel_initializer, arc_b, arc_t, adapt_actfun=False):
        """ parameters are dictionaries """
        super().__init__()
        self.layer_b = layers["branch"]
        self.layer_t = layers["trunk"]
        self.act_b   = activation_str["branch"]
        self.act_t   = activation_str["trunk"]
        self.init_b  = kernel_initializer["branch"]
        self.init_t  = kernel_initializer["trunk"]
        self.arc_b   = arc_b
        self.arc_t   = arc_t
        self.adapt   = adapt_actfun
        
        if self.arc_b == "FNN":
            self.branch  = FNN(self.layer_b, self.act_b, self.init_b, self.adapt)
        elif arc_b == "FNN_LN":
            self.branch  = FNN_LN(self.layer_b, self.act_b, self.init_b, self.adapt)
        else:
            raise NotImplementedError("Architecture for branch not implemented yet.")

        if self.arc_t == "FNN":
            self.trunk   = FNN(self.layer_t, self.act_t, self.init_t, self.adapt)
        elif arc_t == "FNN_LN":
            self.trunk  = FNN_LN(self.layer_t, self.act_t, self.init_t, self.adapt)
        else:
            raise NotImplementedError("Architecture for trunk not implemented yet.")
            
        # Final bias
        self.b = nn.parameter.Parameter(torch.tensor(0.0))

    def forward(self,x):
        b_in = x[0]
        t_in = x[1]
        b_in = self.branch(b_in)
        # Notice that in trunk we apply the activation
        # also to the last layer
        t_in = activation(self.act_t)(self.trunk(t_in))
        #print(b_in.shape, t_in.shape)
        out = torch.einsum("ij,kj->ik",b_in,t_in) # check with dataset
        # add bias
        out += self.b
        return out