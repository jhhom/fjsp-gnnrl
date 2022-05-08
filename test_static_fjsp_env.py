from static_fjsp_env.fjsp_env import FJSP
import numpy as np

jobs = np.array([[[0, 2, 1],
        [0, 3, 1],
        [1, 0, 2]],

       [[1, 1, 0],
        [0, 2, 2],
        [0, 3, 3]],

       [[3, 0, 0],
        [3, 0, 0],
        [0, 0, 0]]], dtype=np.int32)

fjsp = FJSP(n_j=3, n_m=3, num_of_operations_ub_per_job=3)
fjsp.reset(jobs, 3)

print(fjsp.operation_end_times)
print(fjsp.lower_bounds)

print()
print()
print()
print('LOWER BOUND LOWER BOUDN')
print()

fjsp.step((0, 1))
print('LB')
print(fjsp.lower_bounds)
