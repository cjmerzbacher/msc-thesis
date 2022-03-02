from torchdiffeq import odeint_adjoint as odeint
import numpy as np 
import torch.nn as nn
import torch.optim as optim
import torch
from functions import activation, repression, nonlinearity

#Define custom Module with fixed dual control architecture
class DualControl(nn.Module):
    def __init__(self):
        '''Constructor instantiating weights as model parameters and constants'''
        super().__init__()
        
        #Initialize constants, taken from Verma et al paper.
        self.Vin = 1.
        self.e0 = 0.0467
        self.lam = 1.93E-4 #1/s
        #Assume equal kinetics for all three enzymes
        self.kc = 12
        self.km = 10 #1/s

        #Initizalize weights for training
        self.W = nn.Parameter(torch.from_numpy(np.array([[2,2],[1,1], [1E-7, 1E-7]]))) 
        #parameters are n1, n2, theta1, theta2, k1, k2
        
    def forward(self, t, y):
        '''Computes derivatives of system of differential equations'''
        ydot = torch.zeros(6)
        ydot[0] = self.Vin - self.lam*y[0] - self.e0*nonlinearity(y[0], self.kc, self.km) - self.lam*y[1]
        ydot[1] = y[2]*nonlinearity(y[0], self.kc, self.km) - y[3]*nonlinearity(y[1], self.kc, self.km) - self.lam*y[1]
        ydot[2] = repression(y[1], self.W[2][0], self.W[1][0], self.W[0][0]) - self.lam*y[2]
        ydot[3] = activation(y[1], self.W[2][1], self.W[1][1], self.W[0][1]) - self.lam*y[3]
        ydot[4] = (self.Vin -  y[3]*nonlinearity(y[1], self.kc, self.km))**2
        ydot[5] = repression(y[1], self.W[2][0], self.W[1][0], self.W[0][0]) + activation(y[1], self.W[2][1], self.W[1][1], self.W[0][1])
        return ydot

#Define custom Module with fixed dual control architecture with parameters equalized
class DualControlEqualParams(nn.Module):
    def __init__(self):
        '''Constructor instantiating weights as model parameters and constants'''
        super().__init__()
        
        #Initialize constants, taken from Verma et al paper.
        self.Vin = 1.
        self.e0 = 0.0467
        self.lam = 10 #1/s
        #Assume equal kinetics for all three enzymes
        self.kc = 12
        self.km = 10 #1/s

        #Initizalize weights for training
        self.W = nn.Parameter(torch.from_numpy(np.array([[2,2],[1,1], [1E-7, 1E-7]]))) 
        #parameters are n1, n2, theta1, theta2, k1, k2
        
    def forward(self, t, y):
        '''Computes derivatives of system of differential equations'''
        ydot = torch.zeros(6)
        ydot[0] = self.Vin - self.lam*y[0] - self.e0*nonlinearity(y[0], self.kc, self.km) - self.lam*y[1]
        ydot[1] = y[2]*nonlinearity(y[0], self.kc, self.km) - y[3]*nonlinearity(y[1], self.kc, self.km) - self.lam*y[1]
        ydot[2] = repression(y[1], self.W[2][0], self.W[1][0], self.W[0][0]) - self.lam*y[2]
        ydot[3] = activation(y[1], self.W[2][1], self.W[1][1], self.W[0][1]) - self.lam*y[3]
        ydot[4] = (self.Vin -  y[3]*nonlinearity(y[1], self.kc, self.km))**2
        ydot[5] = repression(y[1], self.W[2][0], self.W[1][0], self.W[0][0]) + activation(y[1], self.W[2][1], self.W[1][1], self.W[0][1])
        return ydot