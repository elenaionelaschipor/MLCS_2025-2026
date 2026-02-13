"""
utility_dataset.py

author: Edoardo Centofanti

Functions for easier dataset manipulation
"""

import torch
import scipy.io as sio

"""
Load dataset
"""
def load_dataset(dataname,full_v_data=False):
    d         = sio.loadmat(dataname)
    var_y     = [key for key in d.keys() if not key.startswith('__') and len(key)<3][0]
    u_data    = torch.tensor(d[var_y]).float()
    x_data    = torch.tensor(d['tspan']).float()
    v_data1   = (torch.tensor(d['iapps'])[:,0:2]).float()   # pulse times
    v_data2   = (torch.tensor(d['iapps'])[:,[2]]).float()   # pulse intensities

    # Variant for v_data
    if full_v_data:
        domain = x_data.flatten().repeat(v_data1.shape[0], x_data.shape[0])
        v_data = torch.where((domain >= v_data1[:, [0]]) & (domain <= v_data1[:, [1]]), 1.0, 0.0)
        v_data = v_data*v_data2
    else:
        v_data = torch.cat((v_data1,v_data2),axis=1)

    return u_data, x_data.t(), v_data