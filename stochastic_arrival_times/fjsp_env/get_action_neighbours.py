import numpy as np

def get_action_neighbours(action_op_id, machine_op_ids):
    action_coordinate = np.where(machine_op_ids == action_op_id)
    preceding = machine_op_ids[
        action_coordinate[0],
        action_coordinate[1] - 1
            if action_coordinate[1].item() > 0
            else action_coordinate[1].item()
    ].item()
    succeding = machine_op_ids[
        action_coordinate[0],
        action_coordinate[1] + 1
            if action_coordinate[1].item() + 1 < machine_op_ids.shape[1]
            else action_coordinate[1]
    ].item()
    succeding = action_op_id if succeding < 0 else succeding
    return int(preceding), int(succeding)


