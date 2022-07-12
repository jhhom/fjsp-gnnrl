from random import choice, randint
import numpy as np

from .problem_configs import datasetConfigs


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
            uniform_instance_gen(
                num_of_jobs=config.num_of_jobs,
                num_of_machines=config.num_of_machines,
                lowest_num_of_operation_per_job=config.lowest_num_of_operations_per_job,
                highest_num_of_operation_per_job=config.highest_num_of_operations_per_job,
                highest_num_of_alternatives_per_op=config.num_of_alternative_bounds[1],
                lowest_num_of_alternatives_per_op=config.num_of_alternative_bounds[0],
                duration_ub=config.duration_bounds[1],
                duration_lb=config.duration_bounds[0],
            ) for _ in range(12)
        ]

        problems = np.array(problems, dtype=np.int32)
        np.save(f'./dataset/problems_dataset/dataset/{size}_12', problems)