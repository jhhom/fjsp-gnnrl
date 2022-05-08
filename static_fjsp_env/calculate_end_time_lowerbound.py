import numpy as np

def last_non_zero(arr, axis, invalid_value=-1):
    array = np.copy(arr)
    array = np.max(arr, axis=2)
    mask = array != 0
    value = array.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    y_axis = np.where(mask.any(axis=axis), value, invalid_value)
    x_axis = np.arange(array.shape[0], dtype=np.int64)
    x_return = x_axis[y_axis >= 0]
    y_return = y_axis[y_axis >= 0]
    return x_return, y_return


def print_matrix(array):
    if len(array.shape) == 3:
        array = np.hstack(tuple(array))
    str = ''
    for i in range(len(array)):
        for j in range(len(array[i])):
            str += '{:>3} '.format(array[i][j] )
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


def calculate_end_time_lowerbound(temp_operation_end_times, job_makespans, jobs, operation_end_times):
    job, operation = last_non_zero(operation_end_times, 1, invalid_value=-1)
    job_makespans[np.where(operation_end_times != 0)] = 0
    job_makespans[job, operation] = operation_end_times[job, operation]
    temp = np.copy(job_makespans)
    temp[np.where(job_makespans == 0)] = 9999
    min = np.amin(temp, axis=2)
    min[np.where(min == 9999)] = 0
    min_sum = np.cumsum(min, axis=1)
    for i in range(len(min_sum[:])):
        for j in range(len(min_sum[i, :])):
            min_sum[j, i] = max(min_sum[j, i], 5 * (i+1))

    shifted_sum = np.roll(min_sum, 1)
    shifted_sum[:, 0] = 0
    shifted_sum = np.repeat(shifted_sum, operation_end_times.shape[2]).reshape(operation_end_times.shape[0], operation_end_times.shape[1], operation_end_times.shape[2])

    lower_bounds = job_makespans + shifted_sum
    lower_bounds[np.where(jobs == 0)] = 0
    lower_bounds[np.where(operation_end_times != 0)] = 0

    return lower_bounds + operation_end_times



if __name__ == '__main__':
    jobs = np.array(
        [
            [
                [1, 3, 2],
                [0, 2, 1],
                [0, 0, 3],
            ],
            [
                [0, 0, 2],
                [2, 2, 1],
                [0, 0, 0]
            ],
            [
                [2, 1, 0],
                [1, 0, 1],
                [3, 0, 0]
            ]
        ], dtype=np.int32)
    operation_end_times = np.array([
        [[2, 2, 2], [7, 7, 7], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    ], dtype=np.int32)
    # temp_operation_end_times is max of "real end times" and (arrival times for next operation)
    
    job_makespans = np.array([
        [[0, 0, 0], [7, 7, 7], [0, 0, 3]],
        [[0, 0, 2], [2, 2, 1], [0, 0, 0]],
        [[2, 1, 0], [1, 0, 1], [3, 0, 0]]
    ], dtype=np.int32)

    result = calculate_end_time_lowerbound(job_makespans, jobs, operation_end_times)
    print('final result')
    print(result)



