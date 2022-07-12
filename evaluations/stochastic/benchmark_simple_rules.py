import numpy as np

from evaluations.standard.boxplot.dispatching_rules import spt, mwkr, lwkr
from stochastic_arrival_times.fjsp_env.fjsp_env import StochasticFJSP

if __name__ == '__main__':
    for problem_size in range(1, 11):
        problem = np.load('./evaluations/standard/brandimarte/brandimarte_dataset_numpy/Mk{:02d}.fjs.npy'.format(problem_size))

        env = StochasticFJSP(
            n_j=problem.shape[0],
            n_m=problem.shape[2],
            num_of_operations_ub_per_job=problem.shape[1])

        release_times = open('./stochastic_arrival_times/brandimarte/MK{:02d}_AR.txt'.format(problem_size)).read().split(' ')
        release_times = [x.replace('\n', '') for x in release_times]
        release_times = [int(x) for x in release_times]
        
        adj, feat, omega, mask, machine_feat = env.reset(problem, ub_num_of_operations_per_job=problem.shape[1], release_times=release_times)
        done = False
        omega = np.array(omega, dtype=np.int32)
        mask = np.array(mask, dtype=bool)
        while not done:
            action = lwkr(omega, mask, env.jobs, env.op_id_to_job_info, env.num_of_ops_for_every_job)
            # action = spt(omega, mask, env.jobs, env.op_id_to_job_info)
            adj, feat, reward, done, omega, mask, machine_feat = env.step(action)
            omega = np.array(omega, dtype=np.int32)
            mask = np.array(mask, dtype=bool)
        print(f'{problem_size}: {env.max_end_time}')

