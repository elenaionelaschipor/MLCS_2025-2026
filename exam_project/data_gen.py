
import FEsolver   # usage: FEsolver.generate_data(num_samples, num_collocation_points)
import torch

num_samples_train = 250 
num_samples_test = 500 
num_points_test = 256


l = 3
num_collocation_points = 2**(l)
# Generate data
print('generating data...')
x_train, a_train, u_train, gradu_train = FEsolver.generate_data(num_samples_train, num_collocation_points)
print('done')
torch.save(x_train, "data/x_train"+"_"+str(num_collocation_points)+"_"+".pt")
torch.save(a_train, "data/a_train"+"_"+str(num_collocation_points)+"_"+".pt")
torch.save(u_train, "data/u_train"+"_"+str(num_collocation_points)+"_"+".pt")
torch.save(gradu_train, "data/gradu_train"+"_"+str(num_collocation_points)+"_"+".pt")
print("trained data with", num_collocation_points, "collocation points per side saved")

x_test, a_test, u_test, gradu_test = FEsolver.generate_data(num_samples_test, num_collocation_points)
torch.save(x_test, "data/x_test"+"_"+str(num_collocation_points)+"_"+".pt")
torch.save(a_test, "data/a_test"+"_"+str(num_collocation_points)+"_"+".pt")
torch.save(u_test, "data/u_test"+"_"+str(num_collocation_points)+"_"+".pt")
torch.save(gradu_test, "data/gradu_test"+"_"+str(num_collocation_points)+"_"+".pt")

import matplotlib.pyplot as plt
import numpy as np
idx = int(np.floor(num_samples_train*np.random.rand()))
print(idx)
plt.plot(x_train, 1e-1*gradu_train[idx, :])
plt.plot(x_train, u_train[idx, :])
