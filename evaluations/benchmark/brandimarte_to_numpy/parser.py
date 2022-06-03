import json
import numpy as np

from dataset_characteristics import characteristics

class Job:
    def __init__(self, num_of_operations: int, operations: 'list[Operation]'):
        self.num_of_operations: int = num_of_operations
        self.operations: list[self.Operation] = operations

    def __str__(self):
        output = {
            'num_of_operations': self.num_of_operations,
            'operations': [
                {
                    'operation_num': op.operation_num,
                    'processing_infos': [
                        {
                            'machine_number': info.machine_number,
                            'duration': info.duration
                        } for info in op.processing_infos
                    ]
                } for op in self.operations
            ]
        }
        return json.dumps(output)

    class Operation:
        def __init__(self, operation_num: int, processing_infos: 'list[ProcessingInfo]'):
            self.operation_num: int = operation_num
            self.processing_infos: list[self.ProcessingInfo] = processing_infos

        def __str__(self):
            self.processing_infos
            str = f'Op {self.operation_num}: {",".join(map(lambda x: x.__str__(), self.processing_infos))}'
            return str

        class ProcessingInfo:
            def __init__(self, machine_number, duration):
                self.machine_number: int = machine_number
                self.duration: int = duration
            
            def __str__(self):
                return f'{self.machine_number} {self.duration}'


def parse_operations(numbers: 'list[int]') -> 'list[Job.Operation]':
    i = 0
    op_counter = 0
    operations: list[Job.Operation] = []
    while i < (len(numbers) - 1):
        op = Job.Operation(operation_num=op_counter, processing_infos=[])
        op_counter += 1
        for j in range(numbers[i]):
            op.processing_infos.append(Job.Operation.ProcessingInfo(numbers[i+1], numbers[i+2]))
            i += 2
        i += 1
        print(op)
        operations.append(op)
    return operations
    
jobs: 'list[Job]' = []

if __name__ == '__main__':
    file_names = [f'Mk0{i}.fjs' for i in range(1, 10)]
    file_names.append('Mk10.fjs')

    for file_name in file_names:
        lineCounter = 0
        jobs_array = []
        jobs = []

        file = open(f'./brandimarte_dataset/{file_name}')

        for line in file.readlines():
            if lineCounter == 0:
                lineCounter += 1
                continue

            line = line.replace('\t', '', 999)
            split = line.split(' ')
            if len(split) == 0 or all(i == '' for i in split):
                break
            
            while '' in split: split.remove('')
            while '\n' in split: split.remove('\n')
            for i in range(len(split)): split[i] = split[i].replace('\n', '')

            job = Job(
                num_of_operations=int(split[0]),
                operations=parse_operations(list(map(lambda x: int(x), split[1:])))
            )
            jobs.append(job)

            lineCounter += 1

        n_m = characteristics[file_name].num_of_machines
        n_j = characteristics[file_name].num_of_jobs
        n_o_ub = characteristics[file_name].num_of_operations_ub

        jobs_array = [[[0 for _ in range(n_m)] for _ in range(n_o_ub)] for _ in range(n_j)]

        '''
        print(f'FILE NAME: {file_name}')
        print(f'CHARACTERISTICS: {n_j} {n_o_ub} {n_m}')
        print(f'ARRAY: {len(jobs_array)} {len(jobs_array[0])} {len(jobs_array[0][0])}')
        print(f'{len(jobs)}')
        '''
        for i in range(len(jobs)):
            for j in range(jobs[i].num_of_operations):
                for m in jobs[i].operations[j].processing_infos:
                    '''
                    print(f'JOB: {i}')
                    print(f'OPERATION: {j}')
                    print(f'MACHINE NUMBER: {m.machine_number - 1}')
                    print(f'DURATION: {m.duration}')
                    print()
                    '''
                    jobs_array[i][j][m.machine_number - 1] = m.duration

        for i, job in enumerate(jobs_array):
            print(f'OPERATION: {i+1}')
            for j, op in enumerate(job):
                print(op)
            print()

        np.save(f'{file_name}.npy', jobs_array)