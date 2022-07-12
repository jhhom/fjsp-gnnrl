import numpy as np
import json

from .fjsp_env import FJSP

env = FJSP(3, 3, 3)

# SETUP
f = open('test_description.json')
test_description = json.load(f)
test_description['problem'] = np.array(test_description['problem'], dtype=np.int32)
for x in range(len(test_description['steps'])):
    test_description['steps'][x]['machine_start_times'] = np.array(test_description['steps'][x]['machine_start_times'], dtype=np.float64)
    test_description['steps'][x]['machine_op_ids'] = np.array(test_description['steps'][x]['machine_op_ids'], dtype=np.float64)

# ACT
env.reset(test_description['problem'], 3)

# ASSERT
assert np.allclose(env.machine_start_times, test_description['steps'][0]['machine_start_times'])

for y, step in enumerate(test_description['steps'][1:]):
    # ACT
    action = (step['ACTION'][0], step['ACTION'][1])
    done, omega, mask = env.step(action)

    # ASSERT
    assert np.allclose(env.machine_start_times, step['machine_start_times']), 'Machine start times should be same'
    assert np.allclose(env.machine_op_ids, step['machine_op_ids']), 'Operations run on each machine should be the same'

