import numpy as np

def step(x, v, F, dt):
    f1 = F(x)*dt
    v = v+f1*dt
    x = x+v*dt
    return x, v

def rad(x):
    d = x[:,None,:]-x[None,:,:]
    d2 = (d*d).sum(-1)
    d2 += np.eye(len(d2))
    return d, d2

def flj(x):
    d, d2 = rad(x)
    scale = 6*(d2**-7 - d2**-4)
    return (scale[:,:,None]*d).sum(1)

x = np.random.random((12,3))*10
v = np.random.standard_normal(x.shape)
dt = 0.003

# checking the stability of the forces at this cfg.
f1 = flj(x)
print("Max force component:", np.abs(f1).max(0))
# 0.3977565633079673
x1, v1 = step(x, v, flj, dt)
# conservation of angular momentum
# Since v + f(x) = v1
# and \sum_a x_a ^ f_a(x) = 0 (zero net torque)
# then \sum_a x_a ^ v1_a = \sum_a x_a ^ v_a + x_a ^ f_a(x)
#       = \sum_a x_a ^ v_a
#
t = np.cross(x, f1)
print("Net torque:", t.sum(0))
# confirming zero net torque
# array([ 0.00000000e+00,  2.22044605e-16, -1.66533454e-16])
# so these three should all be equivalent:
print("Initial angular momentum:", np.cross(x, v).sum(0))
#array([-22.25065391,  -8.43893058,   6.02670281])
print("Next angular momentum:", np.cross(x1, v1).sum(0))
#array([-22.25065391,  -8.43893058,   6.02670281])
print("Forward-looking angular momentum:", np.cross(x, v1).sum(0))
#array([-22.25065391,  -8.43893058,   6.02670281])

# Re-doing the timestepping algorithm with fixed precision
def to_fixed(x, decimal):
    # decimal = 0, 1, , 62
    return (x * 2**decimal).astype(np.int64)

def from_fixed(x, decimal):
    # decimal = 0, 1, , 62
    return x.astype(float) * 2**-decimal

def fix_step(x_d, v_d, F, dt, decimal):
    # Here, x_d and v_d are fixed precision,
    # while dt is float and
    # F operates on floating point numbers as usual.
    f1 = F(from_fixed(x_d, decimal))
    v_d = v_d + to_fixed(f1*dt, decimal)
    x_d = x_d + (v_d*dt).astype(x_d.dtype)
    return x_d, v_d

def fix_step_r(x_d, v_d, F, dt, decimal):
    # Exact reverse of the fixed_step map
    x_d = x_d - (v_d*dt).astype(x_d.dtype)
    f1 = F(from_fixed(x_d, decimal))
    v_d = v_d - to_fixed(f1*dt, decimal)
    return x_d, v_d

prec = 40 # 12 decimal places
x_d = to_fixed(x, prec)
v_d = to_fixed(v, prec)
x1_d, v1_d = fix_step(x_d, v_d, flj, dt, prec)
nstep = 100 # run 100 more MD steps
for i in range(nstep):
    x1_d, v1_d = fix_step(x1_d, v1_d, flj, dt, prec)
for i in range(nstep): # then reverse them all
    x1_d, v1_d = fix_step_r(x1_d, v1_d, flj, dt, prec)
x0_d, v0_d = fix_step_r(x1_d, v1_d, flj, dt, prec)
print("Time-reversal error (x_d, v_d):")
print( np.abs(x_d-x0_d).max(), np.abs(v_d-v0_d).max() )
