# solve with OR Tools
import numpy as np
import pandas as pd

from ortools_solver.evaluate import flexible_jobshop, convert_numpy_dataset_to_jobs_array

problem_sizes = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10'] # only 01 to 10


for problem_size in problem_sizes[1:]:
    problems = np.load(f'./dataset/problems_dataset/dataset/MK{problem_size}_12.npy')
    solver_results = []
    for i, problem in enumerate(problems):
        jobs = convert_numpy_dataset_to_jobs_array(problem)
        results, makespan, status, duration, num_of_branches = flexible_jobshop(jobs, problem.shape[2])
        solver_results.append({
            'problem': i,
            'makespan': makespan,
            'status': status,
            'duration': duration,
        })

        df = pd.DataFrame(solver_results)
    df.to_csv(f'./dataset/problems_dataset/ortools_solutions/RESULTS_MK{problem_size}_12.csv')

    