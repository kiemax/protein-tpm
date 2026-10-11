import numpy as np


# dummy weight function F for sampling
def F_exclude_near(d, r0=2, base=lambda x: 10 - 0.8 * x):
    # d is your outside distance field (>=0)
    # weight = 0 for d < r0; otherwise use any base profile on (d - r0)
    return np.where(d >= r0, base(d - r0), 0.0)
