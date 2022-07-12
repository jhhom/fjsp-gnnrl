import numpy as np
import json
from evaluations.standard.boxplot.gnn import new_model
from validate import validate
from params import config
from uniform_instance_gen import datasetConfigs


weights = json.load(open('evaluations/weights.json'))['standard']
'''
models = [
    new_model(problem_size=int(k[2:4]), weight_id=v)
    for k, v in weights.items()
]
'''

i = 9
model = new_model(problem_size=i+1, weight_id=weights['MK{:02d}'.format(i+1)])
problem = np.load('./evaluations/standard/brandimarte/brandimarte_dataset_numpy/MK{:02d}.fjs.npy'.format(i+1))
# model = models[i]
ub_ops = datasetConfigs["MK{:02d}".format(i+1)].highest_num_of_operations_per_job
print(ub_ops)
makespan = validate(validation_set=[problem], model=model, ub_num_of_operations_per_job=ub_ops, release_times=None)
print(makespan)

'''
MK01: 49
MK02: 48
MK03: 270
MK04: 103
MK05: 220
MK06: 132
MK07: 235
MK08: 579
MK09: 434
MK10: 404
'''