from .insert_operation import calculate_job_and_machine_ready_times_of_action
import numpy as np

def get_candidate_machine_features(
    omega: list[tuple[int, int]],
    jobs,
    machine_start_times,
    machine_op_ids,
    op_id_to_job_info,
    machines_workload: list[int],
    current_makespan: int,
    mask: list[bool],
):
    job_and_job_ops: list[tuple[int, int]] = [op_id_to_job_info[i] for i, _ in omega]

    candidate_job_machine_ready_times = [
        calculate_job_and_machine_ready_times_of_action(
            action_op=job_and_job_ops[i][1],
            action_machine=omega[i][1],
            action_op_id=omega[i][0],
            jobs=jobs,
            machine_start_times=machine_start_times,
            machine_op_ids=machine_op_ids,
            op_id_to_job_info=op_id_to_job_info,
        ) for i in range(len(omega))
    ]

    # candidate_job_wait_times = [machine - job for job, machine in candidate_job_machine_ready_times]
    candidate_job_wait_times = \
        calculate_job_wait_times(
            mask,
            omega,
            candidate_job_machine_ready_times,
            op_id_to_job_info,
            machine_start_times,
            machine_op_ids,
            jobs
        )

    candidate_machine_workloads = [machines_workload[action_machine] for _, action_machine in omega]
    candidate_machine_idle_times = [current_makespan - workload for workload in candidate_machine_workloads]
    is_candidate_legal = [0 if masked else 1 for masked in mask]

    return np.column_stack((
        is_candidate_legal,
        candidate_job_wait_times,
        candidate_machine_workloads,
        candidate_machine_idle_times
    ))


def calculate_job_wait_times(
    mask,
    omega,
    candidate_job_machine_ready_times,
    op_id_to_job_info,
    machine_start_times,
    machine_op_ids,
    jobs
):
    job_wait_times = []
    for i in range(len(omega)):
        if mask[i]:
            job_wait_times.append(0)
            continue
        else:
            job_ready_time, machine_ready_time = candidate_job_machine_ready_times[i]
            initial_wait_time = machine_ready_time - job_ready_time
            initial_wait_time = 0 if initial_wait_time < 0 else initial_wait_time
            possible_positions = np.where(job_ready_time < machine_start_times[omega[i][1]])[0]
            if len(possible_positions) == 0:
                job_wait_times.append(initial_wait_time)
                continue
            else:
                action_machine = omega[i][1]
                start_times_of_possible_positions = machine_start_times[action_machine][possible_positions]

                op_ids_of_possible_positions = machine_op_ids[action_machine][possible_positions]
                operations_of_possible_positions = [op_id_to_job_info[int(i)] for i in op_ids_of_possible_positions]
                durations_of_possible_positions = [
                    jobs[x[0]][x[1]][action_machine] for x in operations_of_possible_positions
                ]
                if possible_positions[0] - 1 < 0:
                    start_time_earliest = max(job_ready_time, 0)
                else:
                    op_id_of_earlist_position = machine_op_ids[action_machine][possible_positions[0] - 1]
                    earliest_job, earliest_op = op_id_to_job_info[int(op_id_of_earlist_position)]
                    # we need to get duration for last action on action_machine
                    start_time_earliest = max(
                        job_ready_time,
                        machine_start_times[action_machine][possible_positions[0] - 1] + jobs[earliest_job][earliest_op][action_machine]
                    )
                end_times_for_possible_pos = np.append(start_time_earliest, (start_times_of_possible_positions + durations_of_possible_positions))[:-1]
                possible_gaps = start_times_of_possible_positions - end_times_for_possible_pos
                action_job, action_op = op_id_to_job_info[omega[i][0]]
                action_duration = jobs[action_job][action_op][action_machine]
                idx_legal_pos = np.where(action_duration <= possible_gaps)[0]
                legal_pos = np.take(possible_positions, idx_legal_pos)
                if len(legal_pos == 0):
                    job_wait_times.append(initial_wait_time)
                else:
                    job_wait_times.append(0)

    return job_wait_times

