import numpy as np

from .insert_operation import insert_operation
from .get_job_info_from_op_id import get_job_info_from_op_id

from .params import params

class FJSP():
    def __init__(self,
        n_j,
        n_m,
        num_of_operations_ub_per_job,
    ):
        self.number_of_jobs = n_j
        self.num_of_machines = n_m
        self.num_of_operations = None
        self.ub_num_of_operations_per_job = num_of_operations_ub_per_job
        self.last_op_id_of_jobs = None
        self.first_op_id_of_jobs = None

        self.jobs = None
        self.initial_quality = None

        self.step_count = 0

        self.partial_solution_sequence = None
        self.dispatched_operation = None
        self.op_id_to_job_info = None

        self.machine_start_times = None
        self.machine_op_ids = None
        self.machine_workload = None

        self.omega = None
        self.mask = None



    def done(self):
        assert self.num_of_operations != None and self.num_of_operations != 0
        return len(self.partial_solution_sequence) == self.num_of_operations

    
    def step(self, action):
        row, col = self.op_id_to_job_info[action[0]]
        is_legal_machine = self.jobs[row][col][action[1]] > 0
        if action[0] not in self.dispatched_operation and is_legal_machine:
            self.step_count += 1

            action_job, action_op = self.op_id_to_job_info[action[0]]
            action_machine = action[1]
            action_op_id = action[0]

            action_duration = self.jobs[action_job, action_op, action_machine]
            self.partial_solution_sequence.append(action)
            self.dispatched_operation.append(action[0])

            action_start_time, is_left_shifted = insert_operation(
                action_op=action_op,
                action_machine=action_machine,
                action_duration=action_duration,
                action_op_id=action_op_id,
                jobs=self.jobs,
                machine_start_times=self.machine_start_times,
                machine_op_ids=self.machine_op_ids,
                op_id_to_job_info=self.op_id_to_job_info,
            )

            if action_op_id not in self.last_op_id_of_jobs:
                indexes_of_job = (action_job * self.num_of_machines) + self.machine_ids
                for i in indexes_of_job:
                    self.omega[i][0] += 1
                    if self.jobs[action_job][action_op + 1][self.omega[i][1]] == 0:
                        self.mask[i] = 1
                    else:
                        self.mask[i] = 0
            else:
                # mask last action
                indexes_of_job = action_job * self.num_of_machines + self.machine_ids
                for i in indexes_of_job:
                    self.mask[i] = 1

        return self.done(), self.omega, self.mask


    def reset(self, data, ub_num_of_operations_per_job):
        self.step_count = 0
        self.jobs = data
        self.number_of_jobs = data.shape[0]
        self.num_of_machines = data.shape[2]
        self.machine_ids = np.array(list(range(self.num_of_machines)), dtype=np.int32)

        reshaped_jobs = self.jobs.reshape(-1, self.jobs.shape[2])
        self.num_of_operations = len(reshaped_jobs) - len(np.where(~reshaped_jobs.any(axis=1))[0])
        self.ub_num_of_operations_per_job = ub_num_of_operations_per_job

        num_of_ops_for_each_job = self.get_number_of_ops_for_every_job(self.jobs)
        self.last_op_id_of_jobs = np.cumsum(num_of_ops_for_each_job) - 1
        self.first_op_id_of_jobs = self.last_op_id_of_jobs - num_of_ops_for_each_job + 1

        self.partial_solution_sequence = []
        self.dispatched_operation = []
        self.op_id_to_job_info = [
            get_job_info_from_op_id(i, self.last_op_id_of_jobs)
            for i in range(self.last_op_id_of_jobs[-1] + 1)
        ]
        self.op_id_to_job_info = np.array(self.op_id_to_job_info, dtype=np.int32)

        machine_info_matrix_shape = (self.num_of_machines, ub_num_of_operations_per_job * self.number_of_jobs)
        self.machine_start_times = -params['duration_ub'] * \
            np.ones(machine_info_matrix_shape)
        self.machine_op_ids = -self.number_of_jobs * \
            np.ones(machine_info_matrix_shape)
        self.machine_workload = [0 for _ in range(self.num_of_machines)]

        self.omega = np.array([[i, j] for i in self.first_op_id_of_jobs for j in range(self.num_of_machines)], dtype=np.int32)
        self.mask = np.full(shape=(len(self.omega)), fill_value=0, dtype=bool)
        self.num_of_ops_for_every_job = self.get_number_of_ops_for_every_job(self.jobs)

        for i in range(len(self.omega)):
            job, op = self.op_id_to_job_info[self.omega[i][0]]
            if self.jobs[job][op][self.omega[i][1]] == 0:
                self.mask[i] = 1

        return self.omega, self.mask

    
    def get_machine_durations(self):
        machine_op_durations = np.copy(self.machine_op_ids)
        for i in range(len(machine_op_durations)):
            for j in range(len(machine_op_durations[i])):
                if machine_op_durations[i][j] < 0:
                    break
                row, col = self.op_id_to_job_info[int(self.machine_op_ids[i][j])]
                machine_op_durations[i][j] = self.jobs[row][col][i]
        return machine_op_durations

    
    def get_makespan(self):
        return (self.machine_start_times + self.get_machine_durations()).max()


    def get_number_of_ops_for_every_job(self, job_matrix):
        return (job_matrix.max(axis=2) != 0).sum(1)

