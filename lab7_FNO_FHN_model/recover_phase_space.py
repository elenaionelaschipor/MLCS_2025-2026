# -*- coding: utf-8 -*-
"""
deepONet_HH_pytorch.py

author: Edoardo Centofanti

Learning FitzHugh Nagumo model with operator learning (DeepONet and FNO)
"""
# internal modules
from src.utility_dataset import *
# external modules
import torch
# for test launcher interface
import matplotlib.pyplot as plt
import numpy as np

idx = torch.tensor([159, 100, 96, 298])
dataset_train_v = "dataset/datasetFHN_train_v.mat"
dataset_train_w = "dataset/datasetFHN_train_w.mat"
dataset_test_v = "dataset/datasetFHN_test_v.mat"
dataset_test_w = "dataset/datasetFHN_test_w.mat"
full_v_data = False
u_train, x_train, v_train = load_dataset(dataset_train_v,full_v_data)
u_test_unscaled, x_test_unscaled, v_test_unscaled = load_dataset(dataset_test_v,full_v_data)
sol_test        = u_test_unscaled[idx]
x_test_unscaled = x_test_unscaled.to('cpu')

u_train_w, _, _ = load_dataset(dataset_train_w,full_v_data)
u_test_unscaled_w, x_test, _ = load_dataset(dataset_test_w,full_v_data)
sol_test_w        = u_test_unscaled_w[idx]

#out_v = np.load("saved_predictions/don_test_107.npy")
#out_w = np.load("saved_predictions/don_test_w_107.npy")
out_v = np.load("saved_predictions/fno_test_102.npy")
out_w = np.load("saved_predictions/fno_test_w_102.npy")
arc   = "FNO"

# Create a figure with 3 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(15, 8))

#  ROW 1: Phase plots (V vs w)
for i in range(4):
    axs[0, i].plot(sol_test[i].to('cpu'), sol_test_w[i].to('cpu'), label='Numerical')
    axs[0, i].plot(out_v[i], out_w[i], 'r--', label=arc + ' approx')

    axs[0, i].set_xlabel('$V$')
    if i == 0:
        axs[0, i].set_ylabel('$w$')

    if i == 3:
        axs[0, i].set_ylim([-0.1, 3])
    else:
        axs[0, i].set_ylim([-0.1, 1.5])

    axs[0, i].grid()
    axs[0, i].legend(loc='upper left')


#  ROW 2: V(t) time series
for i in range(4):
    axs[1, i].plot(sol_test[i].to('cpu'), label='Numerical $V(t)$')
    axs[1, i].plot(out_v[i], 'r--', label=arc + ' approx $V(t)$')

    axs[1, i].set_xlabel('Time index')
    if i == 0:
        axs[1, i].set_ylabel('$V(t)$')

    axs[1, i].grid()
    axs[1, i].legend(loc='upper right')


#  ROW 3: w(t) time series
for i in range(4):
    axs[2, i].plot(sol_test_w[i].to('cpu'), label='Numerical $w(t)$')
    axs[2, i].plot(out_w[i], 'r--', label=arc + ' approx $w(t)$')

    axs[2, i].set_xlabel('Time index')
    if i == 0:
        axs[2, i].set_ylabel('$w(t)$')

    axs[2, i].grid()
    axs[2, i].legend(loc='upper right')


# Tight layout
plt.tight_layout()
plt.show()