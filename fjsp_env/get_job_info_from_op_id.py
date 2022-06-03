
'''
Params
    - op_id - int of range (0, n), n is the number of operations
        - Example: 7, which means the 8th operation (or the 8th node)
    - last_op_id_of_jobs - int array of the id of last operation of every job
        - Example: [2, 5, 7]
            - Job 1 last operation is Op 2
            - Job 2 last operation is Op 5
            - Job 3 last operation is Op 7
    
Example input outputs

- Example 1
    - Input: (1, [2, 5, 7])
    - Output: (0, 1)

- Example 2
    - Input: (5, [2, 5, 7])
    - Output: (1, 0)
'''
def get_job_info_from_op_id(op_id: int, last_op_id_of_jobs) -> 'tuple[int, int]':
    for i in range(len(last_op_id_of_jobs)):
        if last_op_id_of_jobs[i] >= op_id:
            if i == 0:
                action_job = 0
                action_op = op_id
                break
            else:
                action_job = i
                action_op = op_id - (last_op_id_of_jobs[i - 1] + 1)
                break
    return (action_job, int(action_op))


