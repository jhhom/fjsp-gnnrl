from dataclasses import dataclass
from testor2 import flexible_jobshop

@dataclass(frozen=True)
class Operation:
    no_of_machines: int
    machine_processing_infos: 'list[dict[str, int]]'

def get_operations(numbers: 'list[int]') -> 'list[Operation]':
    counter = 0
    operations: list[Operation] = []
    while (counter < len(numbers) - 1):
        operation = Operation(no_of_machines=numbers[counter], machine_processing_infos=[])
        for i in range(numbers[counter]):
            operation.machine_processing_infos.append({
                "machine_no": numbers[counter+1],
                "processing_time": numbers[counter+2]
            })
            counter += 2
        counter += 1
        operations.append(operation)
    return operations


mk10 = open('mk10.txt')

problem_info = {
    "no_of_jobs": 0,
    "no_of_machines": 0,
    "average_no_of_eligible_machines_per_job": 0
}

jobs = []

lineCounter = 0
for line in mk10.readlines():
    split = line.split(' ')
    lineCounter += 1
    if lineCounter == 1:
        problem_info["no_of_jobs"] = int(split[0])
        problem_info["no_of_machines"] = int(split[1])
        problem_info["average_no_of_eligible_machines_per_job"] = int(split[2])
        continue
    job_info = {
        "no_of_operations": 0,
        "operations": [],
    }
    job_info["no_of_operations"] = int(split[0])
    job_info["operations"] = get_operations(list(map(lambda x: int(x), split[1:])))
    jobs.append(job_info)


formatted = []
for j in jobs:
    ops = []
    for op in j['operations']:
        op_mpis = []
        for mpi in op.machine_processing_infos:
            op_mpis.append((mpi['processing_time'], mpi['machine_no']))
        ops.append(op_mpis)
    formatted.append(ops)

# print(formatted, 15)
flexible_jobshop(formatted, 15)
