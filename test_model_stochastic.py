import numpy as np
import json
from evaluations.stochastic.benchmark import new_model, load_benchmark_release_times
from validate import validate
from uniform_instance_gen import datasetConfigs


# Change this, either 1 to 10 based on Brandimarte problems (MK01 to MK10)
PROBLEM_SIZE = 1

if PROBLEM_SIZE > 10 or PROBLEM_SIZE < 1:
    raise Exception('PROBLEM_SIZE must be between 1 and 10')



# 1. Load weight
i = PROBLEM_SIZE - 1
weights = json.load(open('saved_weights.json'))['stochastic']
model = new_model(problem_size=i+1, weight_id=weights['MK{:02d}'.format(i+1)])


# 2. Load benchmark problem
problem = np.load('./evaluations/standard/brandimarte/brandimarte_dataset_numpy/MK{:02d}.fjs.npy'.format(i+1))
ub_ops = datasetConfigs["MK{:02d}".format(i+1)].highest_num_of_operations_per_job

release_times = load_benchmark_release_times(PROBLEM_SIZE)

# 3. Get makespan
makespan = -validate(validation_set=[problem], model=model, ub_num_of_operations_per_job=ub_ops, release_times=[release_times])

print(makespan)

