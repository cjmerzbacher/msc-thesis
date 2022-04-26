import numpy as np
from scikits.odes.ode import ode
import matplotlib.pyplot as plt
import random

### HELPER FUNCTIONS ###############################################
def activation(x, k, theta, n):
    return (k*(x/theta)**n)/(1+(x/theta)**n)

def repression(x, k, theta, n):
    return k/(1+(x/theta)**n)

def nonlinearity(x, kc, km):
    return (kc*x)/(km+x)


def generate_equation(X, E, T, A, W):
    '''Generate genetic equations from architecture matrix'''
    def generated_equation(t, y, ydot):
        '''Generate function to solve with Scikit-ODEs'''
        kc=12.; km=10.; lam=1.93E-4; Vin=1.; e0=0.0467
        ydot[0] = Vin - lam*y[0] - e0*nonlinearity(y[0], kc, km) - y[2]*nonlinearity(y[0], kc, km)
        ydot[1] = y[2]*nonlinearity(y[0], kc, km) - y[3]*nonlinearity(y[1], kc, km) - lam*y[1]
        for e in range(E):
            ydot[e+X] = -lam*y[e+X] + np.sum([np.sum(A[t][e]*[activation(y[T[t]], W[t][e][2], W[t][e][1], W[t][e][0]), repression(y[T[t]], W[t][e][2], W[t][e][1], W[t][e][0]), W[t][e][2]]) for t in range(len(T))])
        ydot[E+X] = (Vin - y[X+1]*nonlinearity(y[X-1], kc, km))**2 #J1
        ydot[E+X+1] = np.sum([np.sum([np.sum(A[t][e]*[activation(y[T[t]], W[t][e][2], W[t][e][1], W[t][e][0]), repression(y[T[t]], W[t][e][2], W[t][e][1], W[t][e][0]), 0]) for t in range(len(T))]) for e in range(E)]) #J2
    return generated_equation

def loss_biological(j1, j2, alpha1, alpha2):
        """Computes scalarized loss including genetic constraint and product production"""
        loss = alpha1*j1 + alpha2*j2
        return j1, j2, loss

def bounds_check(params):
    """Computes bounds check with explicitly defined size"""
    n1, n2 = params[0][:, 0]
    theta1, theta2 = params[0][:, 1]
    k1, k2 = params[0][:, 2]
    if k1 <= 1E-3 and k2 <= 1E-3 and k1 >= 1E-7 and k2 >= 1E-7:
        if n1 <= 4 and n2 <= 4 and n1 >= 1 and n2 >= 1:
            if theta1 >= 0.001 and theta2 >= 0.001 and theta1 <= 10 and theta2 <= 10:
                return True
    else: return False

def generate_initial_params(N):
    #Set random seed
    np.random.seed(2022)
    ics = []
    for i in range(N):
        k1 = random.uniform(1E-7,1E-3)
        k2 = random.uniform(1E-7,1E-3)
        n1 = random.uniform(1,4)
        n2 = random.uniform(1,4)
        theta1 = random.uniform(0.001,10)
        theta2 = random.uniform(0.001,10)

        init_conds = np.array([[[n1, theta1, k1], [n2, theta2, k2]]])
        ics.append(init_conds)
    return ics

def solve_single(X, E, T, A, W, times, y0):
    '''Solve a single iteration of a given architecture's equation'''
    ode_function = generate_equation(X, E, T, A, W)


    solution = ode('cvode', ode_function, old_api=False).solve(times, y0);

    j1, j2 = solution.values.y[-1, -2:]
    j1, j2, loss = loss_biological(j1, j2, alpha1=1E-5, alpha2= 1E-2)
    return loss

def solve_patch(X, E, T, A, params, times, y0, patch_size, step_sizes):
    center_loss = solve_single(X, E, T, A, params, times, y0)
    #Sample randomly from hypersphere
    normal_deviates = np.random.normal(size=(patch_size, params.shape[0], params.shape[1], params.shape[2]))
    radius = np.sqrt((normal_deviates**2).sum(axis=0))
    points = normal_deviates/radius
    scaled_points = step_sizes*points + params
    
    min_loss = center_loss
    new_params = params
    for scaled_params in scaled_points:         
        loss = solve_single(X, E, T, A, scaled_params, times, y0)
        if bounds_check(scaled_params):
            if loss < min_loss:
                new_params = scaled_params
                min_loss = loss
    return new_params, min_loss

def solve_architecture(X, E, T, A, times, y0, num_initializations=1, num_epochs = 100, tolerance=0.0004, patch_size=100, step_sizes= np.array([[[0.1, 0.01, 0.001], [0.1, 0.01, 0.0001]]])):
    initial_params = generate_initial_params(num_initializations)
    for i in range(len(initial_params)):
        print('Descent', i+1, 'of', len(initial_params))
        init_conds = initial_params[i]
        next_params, loss = solve_patch(X, E, T, A, init_conds, times, y0, patch_size, step_sizes)
        losses = [loss]; param_trace = [next_params]

        for i in range(num_epochs):
            new_params, loss = solve_patch(X, E, T, A, next_params,times, y0,patch_size, step_sizes)
            next_params = new_params

            print('Epoch: ', i, 'Loss: ', loss)
            losses.append(loss)
            param_trace.append(new_params)
            
            #Early stopping criterion
            if losses[-2] - losses[-1] < tolerance:
                print('Terminating early at step', i, 'of ', num_epochs)
                break
    return param_trace, losses



