from static_fjsp_env.fjsp_env import FJSP
import numpy as np

def print_matrix(array):
    if len(array.shape) == 3:
        array = np.hstack(tuple(array)).astype(np.int32)
    str = ''
    for i in range(len(array)):
        for j in range(len(array[i])):
            str += '{:>2} '.format(array[i][j] )
            if (j+1) % 3 == 0:
                str += '   ' 
        str += '\n'
    print(str)


def print_matrix_with_format(array):
    if len(array.shape) == 3:
        array = np.hstack(tuple(array))
    str = ''
    for i in range(len(array)):
        for j in range(len(array[i])):
            str += '{:>4} '.format(array[i][j])
            if (j+1) % 3 == 0:
                str += '   ' 
        str += '\n'
    print(str)

jobs = np.array([[[0, 2, 6],
        [0, 3, 7],
        [1, 0, 2]],

       [[1, 1, 0],
        [0, 2, 2],
        [0, 3, 3]],

       [[3, 0, 0],
        [3, 0, 0],
        [0, 0, 0]]], dtype=np.int32)

fjsp = FJSP(n_j=3, n_m=3, num_of_operations_ub_per_job=3)
fjsp.reset(jobs, 3)

print('INITIAL\n')
print_matrix(fjsp.lower_bounds)

_ , _, reward, _, _, _, _ = fjsp.step((0, 1))
print('LB 1\n')
print_matrix(fjsp.lower_bounds)
print(reward)

_ , _, reward, _, _, _, _ = fjsp.step((6, 0))
print('LB 2\n')
print_matrix(fjsp.lower_bounds)
print(reward)

_ , _, reward, _, _, _, _ = fjsp.step((7, 0))
print('LB 3\n')
print_matrix(fjsp.lower_bounds)
print(reward)

_ , _, reward, _, _, _, _ = fjsp.step((3, 0))
print('LB 4\n')
print_matrix(fjsp.lower_bounds)
print(reward)

_ , _, reward, _, _, _, _ = fjsp.step((1, 2))
print('LB 5\n')
print_matrix(fjsp.lower_bounds)
print(reward)