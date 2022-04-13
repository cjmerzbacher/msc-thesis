

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

