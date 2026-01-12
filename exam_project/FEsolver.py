# WARNING: è 2d

from mpi4py import MPI
from dolfinx import mesh
import ufl

from dolfinx.fem import functionspace

from dolfinx.fem.petsc import LinearProblem  
from dolfinx import fem
from dolfinx import default_scalar_type
import numpy as np

import torch
torch.set_default_dtype(torch.float64)


def generate_data(nsamples, nx):   
    # number of samples,nx = number of collocation points per side

    npoints = nx * nx 
    uh = torch.zeros((nsamples, npoints)) 
    graduh = torch.zeros((nsamples, npoints, 2)) 
    a = torch.zeros((nsamples, npoints))
    p = 20


    for i in range(nsamples):           # per ogni campione...
        
        y = 2 * np.random.rand(p) - 1       # coefficienti casuali in [-1,1]   
        points, this_uh, this_a, this_graduh = poisson_solver(y, nx)  # ci fidiamo
        
        normal = np.max(np.abs(this_uh.x.array)) 
        uh[i, :] = torch.tensor(this_uh.x.array / normal) 
        graduh[i, :, :] = torch.tensor(this_graduh.x.array / normal) 
        a[i, :] = torch.tensor(this_a.x.array * normal)

    points = torch.tensor(points)
    return points, a, uh, graduh


def poisson_solver(coefs, nx):   
    # coefs è una lista di 3 coefficienti, sarebbero [a1 a2 a3] usati per definire a(x)
    # nx = numero di punti di collocazione
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx-1, nx-1)
    V =functionspace(domain, ("Lagrange", 1))
    uD = fem.Constant(domain, default_scalar_type(0))

    # Create facet to cell connectivity required to determine boundary facets
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(uD, boundary_dofs, V)

    diff_a = fem.Function(V)
    
    
    def a_fun(x):
        X = x[0]
        Y = x[1]

        z = np.ones_like(X)  # a0 = 1

        for i in range(p):
            idx = i + 1  # per usare i = 1,...,p

            m = int(np.floor((idx + 2)/2))
            n = int(np.ceil((idx + 2)/2))

            amplitude = 0.1 * idx**-2
            z += y[i] * amplitude * np.sin(np.pi * m * X) * np.sin(np.pi * n * Y)

        return z 


    diff_a.interpolate(a_fun)

    # forma variazionale

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, 1.0)   # ora lo facciamo costante FEM
    a_form = ufl.dot(diff_a * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_form = f * v * ufl.dx


    a = ufl.dot(diff_a * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc],
                                    petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    uh = problem.solve()

    grad_expr = fem.Expression(ufl.grad(uh), V.element.interpolation_points())
    graduh = fem.Function(V)
    graduh.interpolate(grad_expr)

    points = V.tabulate_dof_coordinates()
    return points, uh, diff_a, graduh


