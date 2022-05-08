import numpy as np

from .params import params

def insert_operation(
    action_job,
    action_op,
    action_machine,
    action_duration,
    action_op_id,
    jobs,
    machine_start_times,
    machine_op_ids,
    last_op_id_of_jobs,
    op_id_to_job_info,
):
    action_job_ready_time, action_machine_ready_time = calculate_job_and_machine_ready_times_of_action(
        action_op, action_machine, action_op_id, jobs=jobs, machine_start_times=machine_start_times, machine_op_ids=machine_op_ids, op_id_to_job_info=op_id_to_job_info
    )
    start_times_for_machine_of_action = machine_start_times[action_machine]
    action_machine_op_ids = machine_op_ids[action_machine]

    is_left_shifted = False
    possible_positions = np.where(action_job_ready_time < start_times_for_machine_of_action)[0]
    if len(possible_positions) == 0:
        action_start_time = put_in_the_end(
            action=action_op_id,
            action_job_ready_time=action_job_ready_time,
            action_machine_ready_time=action_machine_ready_time,
            start_times_for_action_machine=start_times_for_machine_of_action,
            op_ids_for_action_machine=action_machine_op_ids,
        )
    else:
        legal_pos_idx, legal_pos, end_times_for_possible_pos = calculate_legal_positions(
            action_duration=action_duration,
            action_machine=action_machine,
            action_job_ready_time=action_job_ready_time,
            jobs=jobs,
            possible_positions=possible_positions,
            start_times_for_action_machine=start_times_for_machine_of_action,
            action_machine_op_ids=action_machine_op_ids,
            op_id_to_job_info=op_id_to_job_info,
        )
        if len(legal_pos) == 0:
            action_start_time = put_in_the_end(
                action=action_op_id,
                action_job_ready_time=action_job_ready_time,
                action_machine_ready_time=action_machine_ready_time,
                start_times_for_action_machine=start_times_for_machine_of_action,
                op_ids_for_action_machine=action_machine_op_ids,
            )
        else:
            is_left_shifted = True
            action_start_time = put_in_between(
                action=action_op_id,
                legal_pos_indexes=legal_pos_idx,
                legal_pos=legal_pos,
                end_times_for_possible_pos=end_times_for_possible_pos,
                start_times_for_action_machine=start_times_for_machine_of_action,
                op_ids_for_action_machine=action_machine_op_ids                
            )
    return action_start_time, is_left_shifted


def calculate_job_and_machine_ready_times_of_action(
    action_op,
    action_machine,
    action_op_id,
    jobs,
    machine_start_times,
    machine_op_ids,
    op_id_to_job_info,
):
    '''
    jobs matrix
        - row: operation
        - column: machine
    '''
    preceding_job_op_id = action_op_id - 1 \
        if action_op != 0 else None
    
    if len(np.where(machine_op_ids[action_machine] >= 0)[0]) != 0:
        preceding_machine_op_id = machine_op_ids \
            [action_machine] \
            [np.where(machine_op_ids[action_machine] >= 0)] \
            [-1]
    else:
        preceding_machine_op_id = None

    action_job_ready_time = 0
    action_machine_ready_time = 0

    if preceding_job_op_id is not None:
        preceding_job, preceding_job_op = op_id_to_job_info[int(preceding_job_op_id)]
        machine_of_preceding_job_op_index = np.column_stack(np.where(machine_op_ids == preceding_job_op_id))[0]
        machine_of_preceding_job_op = machine_of_preceding_job_op_index[0]
        position_of_preceding_job_op_in_machine = machine_of_preceding_job_op_index[1]
        duration_of_preceding_job_op = jobs[preceding_job][preceding_job_op][machine_of_preceding_job_op]
        action_job_ready_time = (
            machine_start_times[machine_of_preceding_job_op][position_of_preceding_job_op_in_machine] + \
            duration_of_preceding_job_op        
        ).item()
    
    if preceding_machine_op_id is not None:
        preceding_machine_job, preceding_machine_op = op_id_to_job_info[int(preceding_machine_op_id)]
        duration_of_preceding_machine_op = jobs[preceding_machine_job][preceding_machine_op][action_machine]
        order_of_preceding_machine_op = np.where(np.isclose(machine_op_ids[action_machine], preceding_machine_op_id))

        action_machine_ready_time = (
            machine_start_times[action_machine][order_of_preceding_machine_op] + \
                duration_of_preceding_machine_op
        ).item()
    
    return action_job_ready_time, action_machine_ready_time


def calculate_legal_positions(
    action_duration,
    action_machine,
    action_job_ready_time,
    jobs,
    possible_positions,
    start_times_for_action_machine,
    action_machine_op_ids,
    op_id_to_job_info,
):
    start_times_of_possible_positions = start_times_for_action_machine[possible_positions]
    '''
    To get duration of possible positions, take these steps:
    1. We need to convert from op ids to the specific operations (Job, Operation, Machine)
    2. Then use that to get the duration from jobs matrices
    '''
    op_ids_of_possible_positions = action_machine_op_ids[possible_positions]
    operations_of_possible_positions = [
        op_id_to_job_info[int(i)] for i in op_ids_of_possible_positions
    ]
    durations_of_possible_positions = [
        jobs[x[0]][x[1]][action_machine] for x in operations_of_possible_positions
    ]
    if possible_positions[0] - 1 < 0:
        start_time_earliest = max(action_job_ready_time, 0)
    else:
        op_id_of_earlist_position = action_machine_op_ids[possible_positions[0] - 1]
        earliest_job, earliest_op = op_id_to_job_info[int(op_id_of_earlist_position)]
        # we need to get duration for last action on action_machine
        start_time_earliest = max(
            action_job_ready_time,
            # ERROR THIS LINE ❌
            start_times_for_action_machine[possible_positions[0] - 1] + jobs[earliest_job][earliest_op][action_machine]
        )
    end_times_for_possible_pos = np.append(start_time_earliest, (start_times_of_possible_positions + durations_of_possible_positions))[:-1]
    possible_gaps = start_times_of_possible_positions - end_times_for_possible_pos
    idx_legal_pos = np.where(action_duration <= possible_gaps)[0]
    legal_pos = np.take(possible_positions, idx_legal_pos)
    return idx_legal_pos, legal_pos, end_times_for_possible_pos


def put_in_between(action, legal_pos_indexes, legal_pos, end_times_for_possible_pos, start_times_for_action_machine, op_ids_for_action_machine):
    earliest_idx = legal_pos_indexes[0]
    earliest_pos = legal_pos[0]
    action_start_time = end_times_for_possible_pos[earliest_idx]
    start_times_for_action_machine[:] = np.insert(start_times_for_action_machine, earliest_pos, action_start_time)[:-1]
    op_ids_for_action_machine[:] = np.insert(op_ids_for_action_machine, earliest_pos, action)[:-1]
    return action_start_time


def put_in_the_end(action, action_job_ready_time, action_machine_ready_time, start_times_for_action_machine, op_ids_for_action_machine):
    idx = np.where(start_times_for_action_machine == -params['duration_ub'])[0][0]
    action_start_time = max(action_job_ready_time, action_machine_ready_time)
    start_times_for_action_machine[idx] = action_start_time
    op_ids_for_action_machine[idx] = action
    return action_start_time

