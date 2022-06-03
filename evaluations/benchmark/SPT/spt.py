# mk01 benchmarking
from fjsp_env.fjsp_env import FJSP
from fjsp_env.get_job_info_from_op_id import get_job_info_from_op_id
from dataset_characteristics import Characteristic, characteristics

import numpy as np
import random



class SPT:
    def __init__(self):
        pass

    def act(self, omega, mask) -> tuple[int, int]:
        legal_ops = np.array(np.take(omega, np.where(mask == False), axis=0), dtype=np.int32)

        if legal_ops.shape[1] != 1:
            legal_ops = legal_ops.squeeze()
        else:
            legal_ops = np.array([legal_ops[0][0]], dtype=np.int32)

        duration_indexes = [
            (get_job_info_from_op_id(i, env.last_op_id_of_jobs), j) for i, j in legal_ops
        ]
        durations = [
            brandimarte_dataset[i[0], i[1], j] for i, j in duration_indexes
        ]
        min_duration = np.amin(durations)
        shortest_indexes = np.where(durations == min_duration)[0]
        selected_index = random.choice(shortest_indexes)

        return legal_ops[selected_index]


mk01 = characteristics['Mk01.fjs']

env = FJSP(mk01.num_of_jobs, mk01.num_of_machines, mk01.num_of_operations_ub)



def evaluate_spt(jobs_array, dataset_characteristic: Characteristic) -> int:
    adj, feat, omega, mask, machine_features = env.reset(jobs_array, dataset_characteristic.num_of_operations_ub)

    spt = SPT()

    makespan = -env.initial_quality

    action = spt.act(omega, mask)
    adj, feat, reward, done, omega, mask, machine_feat = env.step(action)

    makespan += reward

    while not done:
        action = spt.act(omega, mask)
        adj, feat, reward, done, omega, mask, machine_feat = env.step(action)
        makespan += reward

    return makespan


if __name__ == '__main__':
    brandimarte_file = 'Mk01.fjs'
    validation_file = 'MK01_validation_set_4.npy'

    brandimarte_dataset = np.load(f'./brandimarte/{brandimarte_file}.npy')
    validations = np.load(f'./validation/{validation_file}')
    print(len(validations))
    print(evaluate_spt(brandimarte_dataset, characteristics[brandimarte_file]))
    for i, validation in enumerate(validations):
        print(f"Validation {i+1}: {evaluate_spt(validation, characteristics[brandimarte_file])}")