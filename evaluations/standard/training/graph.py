import json
import numpy as np
import matplotlib.pyplot as plt

path_to_weight_configs = './evaluations/weights.json'

datapoints = tuple[np.ndarray, np.ndarray]

def load_points(path_to_weight_configs) -> tuple[list[datapoints], list[datapoints]]:
    '''
    Return a tuple (training, stochastic)
    Each element is list of length 10 (MK01-MK10) that contains tuples
    Each tuple is a [training datapoints, validation datapoints]

    Training datapoints is of length 10,000
    Validation datapoints is of length 100
    '''
    weights = json.load(open(path_to_weight_configs))
    weights_standard = weights['standard']
    weights_stochastic = weights['stochastic']

    standard = [0 for _ in range(10)]
    stochastic = [0 for _ in range(10)]
    for size, id in weights_standard.items():
        size = size[2:4]
        standard[int(size) - 1] = (
            -1 * np.array(eval(open(f'./records/MK{size}/ID_{id}/training_log.txt').read()), dtype=np.float32),
            np.array(eval(open(f'./records/MK{size}/ID_{id}/validation_log.txt').read()), dtype=np.float32)
        )

    for size, id in weights_stochastic.items():
        size = size[2:4]
        stochastic[int(size) - 1] = (
            -1 * np.array(eval(open(f'./stochastic_records/MK{size}/ID_{id}/training_log.txt').read()), dtype=np.float32),
            np.array(eval(open(f'./stochastic_records/MK{size}/ID_{id}/validation_log.txt').read()), dtype=np.float32)
        )

    return (standard, stochastic)


def plot_single(points_every_problem, folder_name='standard'):
    for i in range(len(points_every_problem)):
        training, validation = points_every_problem[i]
        y_training = np.mean(training[:, 1].reshape(-1, 50), axis=1)
        y_validation = validation
        x_training = np.arange(0, 10000).reshape(200, -1)[:, 0]
        x_validation = np.arange(0, 10000).reshape(100, -1)[:, 0]
        plt.xlabel('Number of episodes')
        plt.ylabel('Makespan')
        plt.plot(x_training, y_training, label='Training')
        plt.plot(x_validation, y_validation, label='Validation')
        plt.title('Graph of makespan vs number of episodes')
        plt.legend()
        plt.savefig('./evaluations/standard/training/figures/' + folder_name + '/MK{:02d}.svg'.format(i+1), format='svg')
        plt.close()


def plot_all(points_every_problem, folder_name='all_standard'):
    figure, axis = plt.subplots(5, 2)

    for i in range(len(points_every_problem)):
        training, validation = points_every_problem[i]
        y_training = np.mean(training[:, 1].reshape(-1, 50), axis=1)
        y_validation = validation
        x_training = np.arange(0, 10000).reshape(200, -1)[:, 0]
        x_validation = np.arange(0, 10000).reshape(100, -1)[:, 0]
        plt.xlabel('Number of episodes')
        plt.ylabel('Makespan')
        plt.plot(x_training, y_training, label='Training')
        plt.plot(x_validation, y_validation, label='Validation')
        plt.title('Graph of makespan vs number of episodes')
        plt.legend()
        plt.savefig('./evaluations/standard/training/figures/{folder_name}/MK{:02d}.svg'.format(i+1), format='svg')
        plt.close()


def plot_double(y1, y2, title1, title2):
    training, validation = y1
    training2, validation2 = y2
    training = np.mean(training[:, 1].reshape(-1, 50), axis=1)
    training2 = np.mean(training2[:, 1].reshape(-1, 50), axis=1)
    x_training = np.arange(0, 10000).reshape(200, -1)[:, 0]
    x_validation = np.arange(0, 10000).reshape(100, -1)[:, 0]

    figure, axis = plt.subplots(1, 2)

    axis[0].plot(x_training, training, label='Training')
    axis[0].plot(x_validation, validation, label='Validation')
    axis[0].set_xlabel('Number of episodes')
    axis[0].set_ylabel('Makespan')
    axis[0].set_title(title1, pad=10)
    axis[0].legend()

    axis[1].plot(x_training, training2, label='Training')
    axis[1].plot(x_validation, validation2, label='Validation')
    axis[1].set_xlabel('Number of episodes')
    axis[1].set_ylabel('Makespan')
    axis[1].set_title(title2, pad=10)
    axis[1].legend()

    figure.set_size_inches(11, 3.5)
    figure.subplots_adjust(bottom=0.2)

    plt.savefig(f'./evaluations/standard/training/figures/pair_standard/{title1}.svg', format="svg")
    plt.close()


'''
MK01 = weights_standard['MK01']

training_logs = eval(open(f'./records/MK01/ID_{MK01}/training_log.txt', 'r').read())
training_logs = np.array(training_logs, dtype=np.float32)

validation_logs = eval(open(f'./records/MK01/ID_{MK01}/validation_log.txt', 'r').read())

vx = np.arange(0, 10000).reshape(100, -1)[:, 0]
vy = np.array(validation_logs, dtype=np.float32)
x = np.array(training_logs[:, 0], dtype=np.int32).reshape(200, -1)[:, 0]
y = -1 * np.mean(training_logs[:, 1].reshape(-1, 50), axis=1)


plt.plot(x, y, label="Training")
plt.plot(vx, vy, label="Validation")
plt.xlabel('Number of episodes')
plt.ylabel('Makespan')
plt.title('Graph of makespan vs number of episodes')
plt.legend()
plt.show()
'''


points = load_points(path_to_weight_configs)
standard = points[0]
stochastic = points[1]

# PLOT DOUBLE GRAPHS

i = 2
title1 = 'MK{:02d}'.format(i+1)
title2 = 'MK{:02d}'.format(i+2)
plot_double(standard[i], standard[i+1], title1, title2)

# plot_single(standard, 'standard')
