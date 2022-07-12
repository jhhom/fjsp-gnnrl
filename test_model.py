import numpy as np
import json
from evaluations.standard.boxplot.gnn import new_model
from validate import validate
from uniform_instance_gen import datasetConfigs


# Change this, either 1 to 10 based on Brandimarte problems (MK01 to MK10)
PROBLEM_SIZE = 10

if PROBLEM_SIZE > 10 or PROBLEM_SIZE < 1:
    raise Exception('PROBLEM_SIZE must be between 1 and 10')



# 1. Load weight
i = PROBLEM_SIZE - 1
weights = json.load(open('saved_weights.json'))['standard']
model = new_model(problem_size=i+1, weight_id=weights['MK{:02d}'.format(i+1)])


# 2. Load benchmark problem
problem = np.load('./evaluations/standard/brandimarte/brandimarte_dataset_numpy/MK{:02d}.fjs.npy'.format(i+1))
ub_ops = datasetConfigs["MK{:02d}".format(i+1)].highest_num_of_operations_per_job


# 3. Get makespan
makespan = -validate(validation_set=[problem], model=model, ub_num_of_operations_per_job=ub_ops, release_times=None)

print(makespan)

