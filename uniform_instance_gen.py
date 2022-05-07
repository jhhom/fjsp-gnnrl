from dataclasses import dataclass
from random import choice, randint
import numpy as np


@dataclass(frozen=True)
class DatasetConfig:
    num_of_jobs: int
    num_of_machines: int
    highest_num_of_operations_per_job: int
    lowest_num_of_operations_per_job: int
    num_of_alternative_bounds: tuple[int, int]
    num_of_operations_to_num_of_jobs: dict[int, int]
    duration_bounds: tuple[int, int]

    def get_total_num_of_operations(self):
        sum = 0
        for o, j in self.num_of_operations_to_num_of_jobs.items():
            sum += o * j
        return sum


datasetConfigs = {
    "MK01": DatasetConfig(
        num_of_jobs=10,
        num_of_machines=6,
        duration_bounds=(1, 6),
        num_of_alternative_bounds=(1, 3),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=6,
        num_of_operations_to_num_of_jobs={
            5: 5,
            6: 5
        }),
    "MK02": DatasetConfig(
        num_of_jobs=10,
        num_of_machines=6,
        duration_bounds=(1, 6),
        num_of_alternative_bounds=(1, 6),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=6,
        num_of_operations_to_num_of_jobs={
            6: 8,
            5: 2
        }),
    "MK03": DatasetConfig(
        num_of_jobs=15,
        num_of_machines=8,
        duration_bounds=(1, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=10,
        num_of_operations_to_num_of_jobs={
            10: 15
        }),
    "MK04": DatasetConfig(
        num_of_jobs=15,
        num_of_machines=8,
        duration_bounds=(1, 9),
        num_of_alternative_bounds=(1, 3),
        lowest_num_of_operations_per_job=3,
        highest_num_of_operations_per_job=9,
        num_of_operations_to_num_of_jobs={
            9: 2,
            8: 1,
            7: 2,
            6: 4,
            5: 3,
            4: 2,
            3: 1,
        }),
    "MK05": DatasetConfig(
        num_of_jobs=15,
        num_of_machines=4,
        duration_bounds=(5, 9),
        num_of_alternative_bounds=(1, 2),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=9,
        num_of_operations_to_num_of_jobs={
            9: 3,
            8: 2,
            7: 5,
            6: 3,
            5: 2
        }),
    "MK06": DatasetConfig(
        num_of_jobs=10,
        num_of_machines=15,
        duration_bounds=(1, 9),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=15,
        highest_num_of_operations_per_job=15,
        num_of_operations_to_num_of_jobs={
            15: 10
        }),
    "MK07": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=5,
        duration_bounds=(1, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=5,
        highest_num_of_operations_per_job=5,
        num_of_operations_to_num_of_jobs={
            5: 15
        }),
    "MK08": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=10,
        duration_bounds=(5, 19),
        num_of_alternative_bounds=(1, 2),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=14,
        num_of_operations_to_num_of_jobs={
            10: 6,
            11: 8,
            12: 3,
            13: 1,
            14: 2
        }),
    "MK09": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=10,
        duration_bounds=(5, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=14,
        num_of_operations_to_num_of_jobs={
            10: 2,
            11: 7,
            12: 3,
            13: 5,
            14: 3
        }),
    "MK10": DatasetConfig(
        num_of_jobs=20,
        num_of_machines=15,
        duration_bounds=(5, 19),
        num_of_alternative_bounds=(1, 5),
        lowest_num_of_operations_per_job=10,
        highest_num_of_operations_per_job=14,
        num_of_operations_to_num_of_jobs={
            10: 2,
            11: 7,
            12: 3,
            13: 5,
            14: 3,
        }),
}



def uniform_instance_gen(
    num_of_jobs,
    num_of_machines,
    lowest_num_of_operation_per_job,
    highest_num_of_operation_per_job,
    lowest_num_of_alternatives_per_op,
    highest_num_of_alternatives_per_op,
    duration_lb,
    duration_ub
):
    jobs = [
        [
            [0 for _ in range(num_of_machines)]
            for _ in range(highest_num_of_operation_per_job)
        ] for _ in range(num_of_jobs)
    ]
    for i in range(num_of_jobs):
        num_of_operation = randint(lowest_num_of_operation_per_job, highest_num_of_operation_per_job)
        for j in range(num_of_operation):
            num_of_alternatives = np.random.randint(lowest_num_of_alternatives_per_op, highest_num_of_alternatives_per_op)
            machine_ids = list(range(num_of_machines))
            for _ in range(num_of_alternatives):
                col_idx = choice(machine_ids)
                machine_ids.remove(col_idx)

                duration = randint(duration_lb, duration_ub)
                jobs[i][j][col_idx] = duration
        
    return np.array(jobs, dtype=np.int32)


def uniform_instance_gen_with_fixed_num_of_operations(
    num_of_jobs,
    num_of_machines,
    highest_num_of_operation_per_job,
    num_of_alternatives_bounds: 'tuple[int, int]',
    num_of_operations_to_num_of_jobs: 'dict[int, int]',
    durations_bounds: 'tuple[int, int]',
):
    jobs = [
        [
            [0 for _ in range(num_of_machines)]
            for _ in range(highest_num_of_operation_per_job)
        ] for _ in range(num_of_jobs)
    ]


    # eg: [(5, [0, 1, 2]), (6, [3, 4, 5])]
    # means Job 0, 1, 2 has duration 5 operations
    num_of_operations_to_job_indexes: 'dict[int, list[int]]' = {
        j: [] for j in num_of_operations_to_num_of_jobs.keys()
    }

    indexes = list(range(num_of_jobs))

    # Assign job indexes for each number of operations
    '''
    1. For each number of operations, get the number of jobs
    2. For the number of jobs, 
    '''
    for num_of_ops, num_of_jobs in num_of_operations_to_num_of_jobs.items():
        for _ in range(num_of_jobs):
            num = choice(indexes)
            indexes.remove(num)
            num_of_operations_to_job_indexes[num_of_ops].append(num)


    # Assign durations to the job array
    '''
    1. For each job, get it's number of operations
    2. For number of operations, assign a random duration, and a random machine
    '''
    for i, v in enumerate(num_of_operations_to_job_indexes):
        for job_index in num_of_operations_to_job_indexes[v]:
            num_of_operations = v
            for j in range(num_of_operations):
                num_of_alternatives = np.random.randint(num_of_alternatives_bounds[0], num_of_alternatives_bounds[1])
                machine_ids = list(range(num_of_machines))
                for _ in range(num_of_alternatives):
                    col_idx = choice(machine_ids)
                    machine_ids.remove(col_idx)

                    duration = randint(durations_bounds[0], durations_bounds[1])
                    jobs[job_index][j][col_idx] = duration

    return np.array(jobs, dtype=np.int32)



def write_dataset_in_brandimarte_format(
    jobs_array,
    lowest_num_of_alternatives,
    highest_num_of_alternatives,
    file_name
):
    f = open(file_name, 'w')
    lines = []

    # for first line
    num_of_jobs = jobs_array.shape[0]
    num_of_machines = jobs_array.shape[2]
    avg_num_of_alternatives = (lowest_num_of_alternatives + highest_num_of_alternatives) / 2
    lines.append(f'{num_of_jobs} {num_of_machines} {int(avg_num_of_alternatives)}\n')

    # for subsequent lines
    for job in jobs_array:
        num_of_ops = job.shape[0] - len(np.where(~job.any(axis=1))[0])
        line = f'{num_of_ops}'
        for row in job:
            num_of_alternatives = np.count_nonzero(row)
            line += f' {num_of_alternatives}'
            indexes = np.nonzero(row)[0]
            for i in indexes: line += f' {i+1} {row[i]}'
        line += '\n'
        lines.append(line)
    f.writelines(lines)
    f.close()






# generate dataset
if __name__ == '__main__':
    for size, config in datasetConfigs.items():
        problems = [
            uniform_instance_gen_with_fixed_num_of_operations(
                durations_bounds=config.duration_bounds,
                highest_num_of_operation_per_job=config.highest_num_of_operations_per_job,
                num_of_alternatives_bounds=config.num_of_alternative_bounds,
                num_of_jobs=config.num_of_jobs,
                num_of_machines=config.num_of_machines,
                num_of_operations_to_num_of_jobs=config.num_of_operations_to_num_of_jobs
            ) for _ in range(12)
        ]
        problems = np.array(problems, dtype=np.int32)
        np.save(f'./validation/{size}_validation_set_4', problems)