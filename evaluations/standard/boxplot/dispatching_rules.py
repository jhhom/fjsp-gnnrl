import numpy as np

def spt(omega, mask, jobs_matrix, op_id_to_job_info):
    """
    Returns action with minimum processing time

    1. Get the legal actions
        - Legal actions are actions that are not masked
    2. Get the duration (or processing time) of each legal operations
    3. Get the action with minimum duration
    4. If there are ties, randomly select one action from the tie

    Parameters
    ----------
    num2 : int
        Second number to add.

    Returns
    -------
    tuple[int, int]
        The first element is the operation id, the second is the machine id
    """
    legals = omega[~mask]
    indices = np.column_stack((op_id_to_job_info[legals[:, 0]], legals[:, 1]))
    durations = jobs_matrix[indices[:, 0], indices[:, 1], indices[:, 2]]
    selection = np.random.choice(np.where(durations == durations.min())[0])
    return legals[selection]


def calculate_remaining_workload(omega, jobs_matrix, op_id_to_job_info, num_of_ops_for_every_job):
    omega_workloads = np.copy(omega)
    omega_job_info = op_id_to_job_info[omega[:, 0]]
    omega_total_ops = num_of_ops_for_every_job[omega_job_info[:, 0]]
    omega_next_ops = omega_job_info[:, 1] + 1
    omega_leftover_ops = [
        np.arange(x[0], x[1], 1) for x in np.vstack((omega_next_ops, omega_total_ops)).transpose()
    ]
    omega_workloads = []
    for i in range(len(omega_job_info[:, 0])):
        workload = 0
        for op in omega_leftover_ops[i]:
            workload += jobs_matrix[omega_job_info[i][0], op].mean()
        omega_workloads.append(workload)
    return np.array(omega_workloads, dtype=np.float32)


def mwkr(omega, mask, jobs_matrix, op_id_to_job_info, num_of_ops_for_every_job):
    """
    Returns action with minimum processing time

    1. Get the remaining workloads for the job of each action
    2. For illegal actions (masked), set to -1
        - this is so that they will not be selected when we get the action with highest workload
            - (remember that the lowest possible duration for an action is 0)
    3. Get the actions with maximum workloads (note that there may be ties)
    4. To break the ties, we select the first action of a random job (since the remaining workloads for actions of same job is the same)

    Parameters
    ----------
    num2 : int
        Second number to add.

    Returns
    -------
    tuple[int, int]
        The first element is the operation id, the second is the machine id
    """
    omega_workloads = calculate_remaining_workload(omega, jobs_matrix, op_id_to_job_info, num_of_ops_for_every_job)
    omega_workloads[mask] = -1
    choices = omega[np.where(omega_workloads == omega_workloads.max())[0]]
    choices_jobs = op_id_to_job_info[choices[:, 0]][:, 0]
    job_selection = np.random.choice(np.unique(choices_jobs))
    selected = choices[choices_jobs == job_selection]
    return selected[np.random.choice(len(selected))]


def lwkr(omega, mask, jobs_matrix, op_id_to_job_info, num_of_ops_for_every_job):
    omega_workloads = calculate_remaining_workload(omega, jobs_matrix, op_id_to_job_info, num_of_ops_for_every_job)
    omega_workloads[mask] = np.Infinity
    choices = omega[np.where(omega_workloads == omega_workloads.min())[0]]
    choices_jobs = op_id_to_job_info[choices[:, 0]][:, 0]
    job_selection = np.random.choice(np.unique(choices_jobs))
    selected = choices[choices_jobs == job_selection]
    return selected[np.random.choice(len(selected))]



if __name__ == '__main__':
    import json
    from fjsp.fjsp_env import FJSP

    env = FJSP(3, 3, 3)

    f = open('./fjsp/test_description.json')
    test_description = json.load(f)
    test_description['problem'] = np.array(test_description['problem'], dtype=np.int32)
    for x in range(len(test_description['steps'])):
        test_description['steps'][x]['machine_start_times'] = np.array(test_description['steps'][x]['machine_start_times'], dtype=np.float64)
        test_description['steps'][x]['machine_op_ids'] = np.array(test_description['steps'][x]['machine_op_ids'], dtype=np.float64)

    omega, mask = env.reset(test_description['problem'], 3)

    # for y, step in enumerate(test_description['steps'][1:]):
    done = False
    while not done:
        # action = mwkr(omega, mask, env.jobs, env.op_id_to_job_info, env.num_of_ops_for_every_job)
        action = spt(omega, mask, env.jobs, env.op_id_to_job_info)
        done, omega, mask = env.step(action)
        print(action)
    print(env.get_makespan())

