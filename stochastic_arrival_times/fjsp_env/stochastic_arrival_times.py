from scipy.stats import poisson
import numpy as np


def calculate_problem_release_times(problem, machine_utilisation, num_of_jobs, num_of_machines):
    nonzeroes = np.count_nonzero(problem, axis=2)
    nonzeroes[nonzeroes == 0] = 1
    avg_pt = np.sum(problem, axis=2) / nonzeroes
    release_rate = np.sum(avg_pt) / (num_of_jobs * num_of_machines * machine_utilisation)
    release_times = poisson.rvs(release_rate, size=num_of_jobs)
    return np.cumsum(release_times)



def calculate_problem_release_times_batch(problems, machine_utilisation, num_of_jobs, num_of_machines):
    nonzeroes = np.count_nonzero(problems, axis=3)
    nonzeroes[nonzeroes == 0] = 1
    avg_pt = np.sum(problems, axis=3) / nonzeroes
    release_rates = np.sum(avg_pt, axis=(0, 2)) / (num_of_jobs * num_of_machines * machine_utilisation)
    release_times = poisson.rvs(release_rates, size=(len(problems), num_of_jobs))
    return np.cumsum(release_times, axis=1)


if __name__ == '__main__':
    for i in range(1, 11):
        path = '../validation/job_durations/MK%02d_validation_set_4.npy' % i
        datasets = np.load(path)
        times = calculate_problem_release_times_batch(datasets, 0.95, len(datasets[0]), len(datasets[0][0][0]))
        np.savetxt('../validation/job_release_times/MK%02d_0.95.txt' % i, times, fmt="%1f")