# -*- coding: utf-8 -*-
"""
deepONet_FHN_pytorch.py

author: Edoardo Centofanti

Learning FitzHugh-Nagumo model with DeepONet and Neural Operator
"""
# internal modules
from src.utility_dataset import *
from src.architectures import get_optimizer, get_loss
from src.don import DeepONet
from src.fno import FNO1d
from src.training import Training
# external modules
import torch
# for test launcher interface
import os
import yaml
import argparse

#########################################
# default value
#########################################
mydevice = torch.device('cpu')
torch.set_default_device(mydevice) # default tensor device
torch.set_default_dtype(torch.float32) # default tensor dtype

# Define command-line arguments
parser = argparse.ArgumentParser(description="Learning FitzHugh-Nagumo model with DeepONet")
parser.add_argument("--config_file", type=str, default="default_params_don.yml", help="Path to the YAML configuration file")
args = parser.parse_args()

# Read the configuration from the specified YAML file
with open(args.config_file, "r") as config_file:
    config = yaml.safe_load(config_file)

param_file_name = os.path.splitext(args.config_file)[0]

# Now, `param_file_name` contains the name without the .json suffix
print("Test name:", param_file_name)

#########################################
# files names to be saved
#########################################
name_log_dir = 'exp_' + param_file_name
name_model = 'model_' + param_file_name 

#########################################
# Hyperparameter
#########################################
arc           = config.get("arc")
dataset_train = config.get("dataset_train")
dataset_test  = config.get("dataset_test")
batch_size    = config.get("batch_size")
full_v_data   = config.get("full_v_data")   # default False
adapt_actfun  = config.get("adapt_actfun")
scheduler     = config.get("scheduler")
Loss          = config.get("Loss")
epochs        = config.get("epochs")
lr            = config.get("lr")
# DeepONet hyper-parameters
u_dim         = config.get("u_dim")
x_dim         = config.get("x_dim")
G_dim         = config.get("G_dim")
inner_layer_b = config.get("inner_layer_b")
inner_layer_t = config.get("inner_layer_t")
activation_b  = config.get("activation_b")
activation_t  = config.get("activation_t")
arc_b         = config.get("arc_b")
arc_t         = config.get("arc_t") 
initial_b     = config.get("initial_b")
initial_t     = config.get("initial_t")
#### WNO hyper-parameters
width = config.get("width")
level = config.get("level")
#### FNO hyper-parameters
d_a            = config.get("d_a")
d_v            = config.get("d_v")
d_u            = config.get("d_u")
L              = config.get("L")
modes          = config.get("modes")
act_fun        = config.get("act_fun")
initialization = config.get("initialization")
arc_fno        = config.get("arc_fno")
x_padding      = config.get("x_padding")
#### Plotting hyper-parameters
show_every = config.get("show_every")

#########################################
#                 MAIN
#########################################
if __name__=="__main__":
    # fix the seed here for reproducibility
    #### SEED METHOD HERE

    #### Network parameters
    layers, activ, init = None, None, None
    
    if arc=="DON":
        layers = {"branch" : [u_dim] + inner_layer_b + [G_dim],
                  "trunk"  : [x_dim] + inner_layer_t + [G_dim] }
        activ  = {"branch" : activation_b,
                  "trunk"  : activation_t}
        init   = {"branch" : initial_b,
                  "trunk"  : initial_t}
    
    u_train, x_train, v_train = load_dataset(dataset_train,full_v_data)
    u_test, x_test, v_test = load_dataset(dataset_test,full_v_data)
    u_train = u_train-1 # to avoid explosion of gradients
    u_test = u_test-1
    #print(v_train.shape, x_train.shape, u_train.shape)    
    # batch loader
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_train, u_train),
                                                batch_size = batch_size, shuffle=True, generator = torch.Generator(device=mydevice))
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(v_test, u_test),
                                              batch_size = batch_size) 
    
    model = None
    if arc=="DON":
        model = DeepONet(layers,activ,init,arc_b,arc_t,adapt_actfun)
    elif arc=="FNO":
        if not full_v_data:
            raise ValueError("full_v_data must be true")
        model = FNO1d(d_a,d_v,d_u,L,modes,act_fun,initialization,arc_fno,x_padding)
    else:
        raise ValueError("This architecture has not been implemented yet.")
    # Count the parameters
    par_tot = sum(p.numel() for p in model.parameters())
    print("Total trainable parameters: ", par_tot)

    optimizer, schedulerName, scheduler = get_optimizer(model,lr,scheduler)

    # Loss function
    myloss = get_loss(Loss)

    trainer = Training(
        model,
        epochs,
        optimizer,
        schedulerName,
        scheduler,
        myloss,
        dataset_test,
        ntrain=u_train.shape[0],
        ntest=u_test.shape[0],
        train_loader=train_loader,
        test_loader=test_loader,
        x_train=x_train,
        x_test=x_test,
	    device=mydevice,
        show_every=show_every
    )

    trainer.train()

    # write here the save method
    # SAVE METHOD HERE
    torch.save(model.state_dict(), "model_"+arc+"_"+str(par_tot)+"_params.pth")
    