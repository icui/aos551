# modified from https://github.com/okada39/pinn_wave

import numpy as np
import tensorflow as tf


c=1
k=2
sd=0.5


def u0(tx):
    t = tx[..., 0, None]
    x = tx[..., 1, None]
    z = k*x - (c*k)*t
    return tf.sin(z) * tf.exp(-(0.5*z/sd)**2)


def du0_dt(tx):
    with tf.GradientTape() as g:
        g.watch(tx)
        u = u0(tx)
    du_dt = g.batch_jacobian(u, tx)[..., 0]
    return du_dt


num_test_samples = 1000
tx = np.zeros([num_test_samples, 2])
tx[:, 1] = np.linspace(-1, 1, num_test_samples)
np.save('u.npy', u0(tf.constant(tx)).numpy())
np.save('v.npy', du0_dt(tf.constant(tx)).numpy())
