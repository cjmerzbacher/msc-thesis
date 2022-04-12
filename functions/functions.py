
import torch
import numpy as np

### HELPER FUNCTIONS ###############################################
def activation(x, k, theta, n):
    return (k*(x/theta)**n)/(1+(x/theta)**n)

def repression(x, k, theta, n):
    return k/(1+(x/theta)**n)

def nonlinearity(x, kc, km):
    return (kc*x)/(km+x)

### LOSS FUNCTIONS ###############################################
def loss_biological(pred, alpha1, alpha2):
    """Computes scalarized loss including genetic constraint and product production"""
    j1 = pred[-1][-2]
    j2 = pred[-1][-1]
    loss = alpha1*j1 + alpha2*j2
    return j1, j2, loss

### SYSTEMS OF EQUATIONS ###############################################
def dual_control_scipy(x, t, n1, n2, theta1, theta2, k1, k2, kc=12, km=10, lam=1.93E-4, Vin=1., e0=0.0467):
    """Defines dual control system of differential equations for use with scipy solvers"""
    x0, x1, e1, e2, j1, j2 = x
    dx0dt = Vin - lam*x0 - e0*nonlinearity(x0, kc, km) - e1*nonlinearity(x0, kc, km) 
    dx1dt = e1*nonlinearity(x0, kc, km) - e2*nonlinearity(x1, kc, km) - lam*x1
    de1dt = repression(x1, k1, theta1, n1) - lam*e1
    de2dt = activation(x1, k2, theta2, n2) - lam*e2
    j1 = (Vin -  e2*nonlinearity(x1, kc, km))**2
    j2 = repression(x1, k1, theta1, n1) + activation(x1, k2, theta2, n2)
    return [dx0dt, dx1dt, de1dt, de2dt, j1, j2] 

def upstream_repression_scipy(x, t, n1, n2, theta1, theta2, k1, k2, kc=12, km=10, lam=1.93E-4, Vin=1., e0=0.0467):
    """Defines upstream_repression system of differential equations for use with scipy solvers"""
    x0, x1, e1, e2, j1, j2 = x
    dx0dt = Vin - lam*x0 - e0*nonlinearity(x0, kc, km) - e1*nonlinearity(x0, kc, km) 
    dx1dt = e1*nonlinearity(x0, kc, km) - e2*nonlinearity(x1, kc, km) - lam*x1
    de1dt = repression(x1, k1, theta1, n1) - lam*e1
    de2dt = k2 - lam*e2
    j1 = (Vin -  e2*nonlinearity(x1, kc, km))**2
    j2 = repression(x1, k1, theta1, n1)
    return [dx0dt, dx1dt, de1dt, de2dt, j1, j2] 


def downstream_activation_scipy(x, t, n1, n2, theta1, theta2, k1, k2, kc=12, km=10, lam=1.93E-4, Vin=1., e0=0.0467):
    """Defines downstream activation system of differential equations for use with scipy solvers"""
    x0, x1, e1, e2, j1, j2 = x
    dx0dt = Vin - lam*x0 - e0*nonlinearity(x0, kc, km) - e1*nonlinearity(x0, kc, km) 
    dx1dt = e1*nonlinearity(x0, kc, km) - e2*nonlinearity(x1, kc, km) - lam*x1
    de1dt = k1 - lam*e1
    de2dt = activation(x1, k2, theta2, n2) - lam*e2
    j1 = (Vin -  e2*nonlinearity(x1, kc, km))**2
    j2 = activation(x1, k2, theta2, n2)
    return [dx0dt, dx1dt, de1dt, de2dt, j1, j2] 

class DualControlSingleParam(torch.nn.Module):
    """Defines dual control system of differential equations for use with TDE solvers"""
    def __init__(self, k1, k2):
        super(DualControlSingleParam, self).__init__()
        #Initialize constants, taken from Verma et al paper.
        self.Vin = 1.
        self.e0 = 0.0467
        self.lam = 1.93E-4 #1/s
        #Assume equal kinetics for all three enzymes
        self.kc = 12
        self.km = 10 #1/s
        self.W = torch.nn.Parameter(torch.tensor([k1, k2]), requires_grad=True) #n, theta, k
        self.n = 2
        self.theta = 0.1

    def forward(self, t, y):
        dx0 = self.Vin - self.lam*y[0] - self.e0*nonlinearity(y[0], self.kc, self.km) - self.lam*y[1]
        dx1 = y[2]*nonlinearity(y[0], self.kc, self.km) - y[3]*nonlinearity(y[1], self.kc, self.km) - self.lam*y[1]
        de1 = repression(y[1], self.W[0], self.theta, self.n) - self.lam*y[2]
        de2 = activation(y[1], self.W[1], self.theta, self.n) - self.lam*y[3]
        j1 = (self.Vin -  y[3]*nonlinearity(y[1], self.kc, self.km))**2
        j2 = repression(y[1], self.W[0], self.theta, self.n) + activation(y[1], self.W[1], self.theta, self.n)
        return torch.stack([dx0, dx1, de1, de2, j1, j2])