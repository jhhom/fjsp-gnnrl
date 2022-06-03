import matplotlib.pyplot as plt
import numpy as np
import ast

problem_size = '01'
record_id = 1

train_log_file = open(f'./records/MK{problem_size}/ID_{record_id}/training_log.txt')
validation_log_file = open(f'./records/MK{problem_size}/ID_{record_id}/validation_log.txt')

validation_log = ast.literal_eval(validation_log_file.read())
train_log = np.array(ast.literal_eval(train_log_file.read()), dtype=np.float32)

ypoints = -1 * np.mean(train_log[:, 1].reshape(-1, 100), axis=1)
xpoints = np.array(range(10000), dtype=np.int32).reshape(100, -1)[:, 0]

figure, axis = plt.subplots(1, 2)

axis[0].plot(xpoints, ypoints)
axis[0].set_xlabel('Number of episodes')
axis[0].set_ylabel('Makespan')
axis[0].set_title('Training makespan averaged over every 100 episodes', pad=10)

xpoints = np.array(range(10000), dtype=np.int32).reshape(100, -1)[:, 0]
ypoints = np.array(validation_log)
axis[1].plot(xpoints, ypoints)
axis[1].set_xlabel('Number of episodes')
axis[1].set_ylabel('Makespan')
axis[1].set_title('Validation makespan averaged evaluated every 100 episodes', pad=10)

figure.set_size_inches(12.5, 3.5)
figure.subplots_adjust(bottom=0.2)

plt.savefig(f'./evaluations/graphs/MK{problem_size}_ID_{record_id}.png')
