import gym
import numpy as np

from .insert_operation import insert_operation
from .get_job_info_from_op_id import get_job_info_from_op_id
from .calculate_end_time_lowerbound import calculate_end_time_lowerbound
from .get_action_neighbours import get_action_neighbours

from .get_candidate_machine_features import get_candidate_machine_features

from .params import params

class FJSP(gym.Env):
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
        self.left_shifted_flags = []
        self.positive_rewards = []

        self.machine_start_times = None
        self.machine_op_ids = None
        self.machine_workload = None

        # all derived state
        self.job_makespans = None
        self.operations_finish_flags = None
        self.operation_end_times = None

        # returned state
        self.adj_matrix = []
        self.omega = None
        self.mask = None
        self.max_end_time = self.initial_quality
        self.lower_bounds = None

        self.get_action_neighbours = get_action_neighbours


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
            self.operations_finish_flags[action_op_id] = (action_machine + 1)
            self.partial_solution_sequence.append(action)
            self.dispatched_operation.append(action[0])

            action_start_time, is_left_shifted = insert_operation(
                action_job=action_job,
                action_op=action_op,
                action_machine=action_machine,
                action_duration=action_duration,
                action_op_id=action_op_id,
                jobs=self.jobs,
                machine_start_times=self.machine_start_times,
                machine_op_ids=self.machine_op_ids,
                last_op_id_of_jobs=self.last_op_id_of_jobs,
                op_id_to_job_info=self.op_id_to_job_info,
            )

            self.left_shifted_flags.append(is_left_shifted)
            # need to loop because it is not numpy array..
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

            self.operation_end_times[action_job, action_op, :] = action_start_time + action_duration
            self.machine_workload[action_machine] += action_duration

            self.lower_bounds = calculate_end_time_lowerbound(self.operation_end_times, self.job_makespans, self.jobs)

            # update adjacency matrix
            preceding_op, succeding_op = self.get_action_neighbours(action_op_id, self.machine_op_ids)
            self.adj_matrix[:,action_op_id] = 0
            self.adj_matrix[action_op_id, action_op_id] = 1

            if action_op_id not in self.first_op_id_of_jobs:
                self.adj_matrix[action_op_id - 1, action_op_id] = 1
            self.adj_matrix[preceding_op, action_op_id] = 1
            self.adj_matrix[action_op_id, succeding_op] = 1

            if is_left_shifted and preceding_op != action_op_id and succeding_op != action_op_id:
                self.adj_matrix[preceding_op, succeding_op] = 0
            
        # prepare for return
        lb_features = self.lower_bounds.reshape(-1, self.num_of_machines)
        lb_features = lb_features[~np.all(lb_features == 0, axis=1)]
        
        feature = np.column_stack(
            (lb_features / params['end_time_normalizing_coefficient'],
                self.operations_finish_flags / params['end_time_normalizing_coefficient']
            ))

        # feature = np.column_stack((lb_features, self.operations_finish_flags))
        reward = - (self.lower_bounds.max() - self.max_end_time)
        if reward == 0:
            reward = params['reward_scale']
            self.positive_rewards += reward
        self.max_end_time = self.lower_bounds.max()

        machine_features = get_candidate_machine_features(
            omega=self.omega,
            jobs=self.jobs,
            machine_start_times=self.machine_start_times,
            machine_op_ids=self.machine_op_ids,
            machines_workload=self.machine_workload,
            current_makespan=self.operation_end_times.max(),
            mask=self.mask,
            op_id_to_job_info=self.op_id_to_job_info,
        )

        return np.array(self.adj_matrix, dtype=np.int32), feature, reward, self.done(), self.omega, self.mask, machine_features


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
        self.left_shifted_flags = []
        self.positive_rewards = 0
        self.op_id_to_job_info = [
            get_job_info_from_op_id(i, self.last_op_id_of_jobs)
            for i in range(self.last_op_id_of_jobs[-1] + 1)
        ]
        self.op_id_to_job_info = np.array(self.op_id_to_job_info, dtype=np.int32)

        machine_info_matrix_shape = (self.number_of_jobs, ub_num_of_operations_per_job * self.number_of_jobs)
        self.machine_start_times = -params['duration_ub'] * \
            np.ones(machine_info_matrix_shape)
        self.machine_op_ids = -self.number_of_jobs * \
            np.ones(machine_info_matrix_shape)
        self.machine_workload = [0 for _ in range(self.num_of_machines)]

        self.job_makespans = np.copy(self.jobs)
        self.operation_end_times = np.zeros(self.jobs.shape)
        self.operations_finish_flags = np.zeros(self.num_of_operations)

        conj_neighbours = np.eye(self.num_of_operations, k=1, dtype=np.single)
        self_loops = np.eye(self.num_of_operations, dtype=np.single)
        self.adj_matrix = conj_neighbours + self_loops
        for i in range(len(self.last_op_id_of_jobs) - 1):
            self.adj_matrix[self.last_op_id_of_jobs[i]][self.last_op_id_of_jobs[i] + 1] = 0
        self.omega = [[i, j] for i in self.first_op_id_of_jobs for j in range(self.num_of_machines)]
        self.mask = np.full(shape=(len(self.omega)), fill_value=0, dtype=bool)

        for i in range(len(self.omega)):
            job, op = self.op_id_to_job_info[self.omega[i][0]]
            if self.jobs[job][op][self.omega[i][1]] == 0:
                self.mask[i] = 1


        temp = np.copy(self.jobs)
        temp[np.where(self.jobs == 0)] = 9999
        min = np.amin(temp, axis=2)
        min[np.where(min == 9999)] = 0
        min_sum = np.cumsum(min, axis=1)
        shifted_sum = np.roll(min_sum, 1)
        shifted_sum[:, 0] = 0
        shifted_sum = np.repeat(shifted_sum, self.jobs.shape[2]).reshape(self.jobs.shape[0], self.jobs.shape[1], self.jobs.shape[2])
        lower_bounds = self.jobs + shifted_sum
        lower_bounds[np.where(self.jobs == 0)] = 0
        self.lower_bounds = lower_bounds

        self.initial_quality = self.lower_bounds.max() if not params['initial_quality_flag'] else 0
        self.max_end_time = self.initial_quality

        lb_features = self.lower_bounds.reshape(-1, self.num_of_machines)
        lb_features = lb_features[~np.all(lb_features == 0, axis=1)]
        feature = np.column_stack(
            (lb_features / params['end_time_normalizing_coefficient'],
                self.operations_finish_flags / params['end_time_normalizing_coefficient']
            ))
        
        machine_features = get_candidate_machine_features(
            omega=self.omega,
            jobs=self.jobs,
            machine_start_times=self.machine_start_times,
            machine_op_ids=self.machine_op_ids,
            machines_workload=self.machine_workload,
            current_makespan=self.operation_end_times.max(),
            mask=self.mask,
            op_id_to_job_info=self.op_id_to_job_info,
        )
        
        return np.array(self.adj_matrix, np.int64), feature, self.omega, self.mask, machine_features


    def get_number_of_ops_for_every_job(self, job_matrix):
        return (job_matrix.max(axis=2) != 0).sum(1)




'''
>>> a
array([[[0, 2, 1],
        [0, 3, 1],
        [1, 0, 2]],

       [[1, 1, 0],
        [0, 2, 2],
        [0, 3, 3]],

       [[3, 0, 0],
        [3, 0, 0],
        [0, 0, 0]]], dtype=int32)
>>> # 3, 3, 2
>>> # 3, 6, 8
>>> # last_op_id = 2, 5, 7
>>> # first_op_id = 0, 3, 6
'''